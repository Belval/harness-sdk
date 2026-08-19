//! Procedural macros for the Strands Agents Rust SDK.
//!
//! Provides `#[tool]`, the Rust counterpart to the TypeScript `tool()` factory.
//! Where TypeScript derives a tool's input schema from a Zod schema at runtime,
//! this macro derives it from the annotated function's signature at compile time,
//! generating a `Tool` implementation struct.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::{
    parse_macro_input, Attribute, Expr, FnArg, ItemFn, Lit, MetaNameValue, Pat, Token, Type,
};

/// Transforms an `async fn` into a Strands agent [`Tool`](strands_agents::Tool).
///
/// The function's doc comment becomes the tool description, and each parameter
/// becomes a property in the generated JSON input schema. A struct named
/// `<FnNameInPascalCase>Tool` is generated with a `new()` constructor.
///
/// # Example
///
/// ```ignore
/// use strands_agents::tool;
///
/// /// Get the current weather for a location.
/// #[tool]
/// async fn get_weather(location: String) -> String {
///     format!("Weather in {location}: 72F, Sunny")
/// }
///
/// // Register with: .tool(GetWeatherTool::new())
/// ```
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as ItemFn);

    let fn_name = &input_fn.sig.ident;
    let fn_name_str = fn_name.to_string();
    let struct_name = format_ident!("{}Tool", to_pascal_case(&fn_name_str));
    let is_async = input_fn.sig.asyncness.is_some();
    let doc_description = extract_doc_comment(&input_fn.attrs);

    // Optional `#[tool(name = "…", description = "…")]` overrides. The tool name
    // defaults to the function identifier and the description to the doc comment.
    let (name_override, description_override) = match parse_tool_attrs(attr) {
        Ok(overrides) => overrides,
        Err(error) => return error.to_compile_error().into(),
    };
    let tool_name = name_override.unwrap_or_else(|| fn_name_str.clone());
    let description = description_override.unwrap_or(doc_description);

    // Collect (name, type-string, is_option) for each real parameter, skipping
    // the framework-context parameter names the caller may include.
    let mut params: Vec<(String, String, bool)> = Vec::new();
    for arg in &input_fn.sig.inputs {
        let FnArg::Typed(pat_type) = arg else {
            continue;
        };
        let Pat::Ident(pat_ident) = &*pat_type.pat else {
            continue;
        };
        let name = pat_ident.ident.to_string();
        if name == "context" || name == "tool_context" || name == "agent" {
            continue;
        }
        let type_string = type_to_string(&pat_type.ty);
        let is_option = type_string.starts_with("Option");
        params.push((name, type_string, is_option));
    }

    let property_inserts: Vec<_> = params
        .iter()
        .map(|(name, type_string, is_option)| {
            let inner = if *is_option {
                option_inner(type_string)
            } else {
                type_string.clone()
            };
            let json_type = json_schema_type(&inner);
            quote! {
                properties.insert(
                    #name.to_string(),
                    ::serde_json::json!({ "type": #json_type }),
                );
            }
        })
        .collect();

    let required: Vec<String> = params
        .iter()
        .filter(|(_, _, is_option)| !is_option)
        .map(|(name, _, _)| name.clone())
        .collect();

    let param_extractions: Vec<_> = params
        .iter()
        .map(|(name, _, is_option)| {
            let ident = format_ident!("{}", name);
            if *is_option {
                quote! {
                    let #ident = context.tool_use.input.get(#name)
                        .and_then(|value| ::serde_json::from_value(value.clone()).ok());
                }
            } else {
                quote! {
                    let #ident = context.tool_use.input.get(#name)
                        .cloned()
                        .ok_or_else(|| ::strands_agents::StrandsError::model(
                            format!("Missing required parameter: {}", #name)
                        ))
                        .and_then(|value| ::serde_json::from_value(value).map_err(|error|
                            ::strands_agents::StrandsError::model_with_source(
                                format!("Invalid value for parameter: {}", #name), error
                            )
                        ))?;
                }
            }
        })
        .collect();

    let param_idents: Vec<_> = params
        .iter()
        .map(|(name, _, _)| format_ident!("{}", name))
        .collect();
    let call = if is_async {
        quote! { #fn_name(#(#param_idents),*).await }
    } else {
        quote! { #fn_name(#(#param_idents),*) }
    };

    let expanded = quote! {
        #input_fn

        #[derive(Clone, Copy, Default)]
        pub struct #struct_name;

        impl #struct_name {
            /// Creates a new instance of this tool.
            pub fn new() -> Self {
                Self
            }
        }

        #[::strands_agents::reexport::async_trait]
        impl ::strands_agents::Tool for #struct_name {
            fn name(&self) -> &str {
                #tool_name
            }

            fn description(&self) -> &str {
                #description
            }

            fn tool_spec(&self) -> ::strands_agents::ToolSpec {
                let mut properties = ::serde_json::Map::new();
                #(#property_inserts)*
                let required: Vec<String> = vec![#(#required.to_string()),*];
                ::strands_agents::ToolSpec {
                    name: #tool_name.to_string(),
                    description: #description.to_string(),
                    input_schema: Some(::serde_json::json!({
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    })),
                    output_schema: None,
                }
            }

            async fn invoke(
                &self,
                context: ::strands_agents::ToolContext,
            ) -> Result<::serde_json::Value, ::strands_agents::StrandsError> {
                #(#param_extractions)*
                let output = #call;
                ::serde_json::to_value(output).map_err(|error|
                    ::strands_agents::StrandsError::model_with_source("failed to serialize tool output", error)
                )
            }
        }
    };

    TokenStream::from(expanded)
}

/// Parses the `#[tool(...)]` attribute arguments, returning optional `name` and
/// `description` string overrides. Accepts `name = "…"` and `description = "…"`
/// in any order; rejects unknown keys and non-string values.
fn parse_tool_attrs(attr: TokenStream) -> syn::Result<(Option<String>, Option<String>)> {
    let mut name = None;
    let mut description = None;
    if attr.is_empty() {
        return Ok((name, description));
    }
    let metas = Punctuated::<MetaNameValue, Token![,]>::parse_terminated.parse(attr)?;
    for meta in metas {
        let value = match &meta.value {
            Expr::Lit(expr_lit) => match &expr_lit.lit {
                Lit::Str(lit_str) => lit_str.value(),
                other => return Err(syn::Error::new_spanned(other, "expected a string literal")),
            },
            other => return Err(syn::Error::new_spanned(other, "expected a string literal")),
        };
        if meta.path.is_ident("name") {
            name = Some(value);
        } else if meta.path.is_ident("description") {
            description = Some(value);
        } else {
            return Err(syn::Error::new_spanned(
                &meta.path,
                "unsupported `#[tool]` argument; expected `name` or `description`",
            ));
        }
    }
    Ok((name, description))
}

// Parameters are deserialized from an owned `serde_json::Value`, so only owned
// types work; borrowed `&str` is intentionally absent (it would not compile).
fn json_schema_type(rust_type: &str) -> &'static str {
    match rust_type {
        "String" | "char" => "string",
        "bool" => "boolean",
        "f32" | "f64" => "number",
        "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "u8" | "u16" | "u32" | "u64" | "u128"
        | "usize" => "integer",
        _ if rust_type.starts_with("Vec") => "array",
        _ => "object",
    }
}

fn option_inner(type_string: &str) -> String {
    type_string
        .strip_prefix("Option<")
        .and_then(|rest| rest.strip_suffix('>'))
        .map(str::to_string)
        .unwrap_or_else(|| type_string.to_string())
}

fn type_to_string(ty: &Type) -> String {
    quote!(#ty).to_string().replace(' ', "")
}

fn extract_doc_comment(attrs: &[Attribute]) -> String {
    let mut lines = Vec::new();
    for attr in attrs {
        if !attr.path().is_ident("doc") {
            continue;
        }
        if let syn::Meta::NameValue(meta) = &attr.meta {
            if let syn::Expr::Lit(expr_lit) = &meta.value {
                if let syn::Lit::Str(lit_str) = &expr_lit.lit {
                    let line = lit_str.value().trim().to_string();
                    let lower = line.to_lowercase();
                    if lower.starts_with("# arg")
                        || lower.starts_with("args:")
                        || lower.starts_with("arguments:")
                    {
                        break;
                    }
                    lines.push(line);
                }
            }
        }
    }
    lines.join(" ").trim().to_string()
}

fn to_pascal_case(input: &str) -> String {
    input
        .split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().chain(chars).collect::<String>(),
                None => String::new(),
            }
        })
        .collect()
}
