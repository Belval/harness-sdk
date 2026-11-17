//! The AWS chain with a profile and files in place of the environment's, shared by the Bedrock
//! and Bedrock Mantle providers.

use std::path::Path;

use aws_config::BehaviorVersion;

/// The AWS chain, with `profile` and the files in place of `AWS_PROFILE`, `AWS_CONFIG_FILE` and
/// `AWS_SHARED_CREDENTIALS_FILE`. A profile also takes the credentials and region from itself, so
/// neither access keys nor `AWS_REGION` in the environment override it.
// `ProfileFiles` is deprecated for `aws_runtime`'s `EnvConfigFiles`, which aws-config's own
// setters still take under this name.
#[allow(deprecated)]
pub(crate) fn aws_loader(
    profile: Option<&str>,
    config_file: Option<&Path>,
    credentials_file: Option<&Path>,
) -> aws_config::ConfigLoader {
    use aws_config::meta::region::RegionProviderChain;
    use aws_config::profile::profile_file::{ProfileFileKind, ProfileFiles};
    use aws_config::profile::{ProfileFileCredentialsProvider, ProfileFileRegionProvider};

    let mut loader = aws_config::defaults(BehaviorVersion::latest());
    let files = (config_file.is_some() || credentials_file.is_some()).then(|| {
        let mut files = ProfileFiles::builder();
        files = match config_file {
            Some(path) => files.with_file(ProfileFileKind::Config, path),
            None => files.include_default_config_file(true),
        };
        files = match credentials_file {
            Some(path) => files.with_file(ProfileFileKind::Credentials, path),
            None => files.include_default_credentials_file(true),
        };
        files.build()
    });
    if let Some(files) = &files {
        loader = loader.profile_files(files.clone());
    }
    if let Some(name) = profile {
        let mut credentials = ProfileFileCredentialsProvider::builder().profile_name(name);
        let mut region = ProfileFileRegionProvider::builder().profile_name(name);
        if let Some(files) = &files {
            credentials = credentials.profile_files(files.clone());
            region = region.profile_files(files.clone());
        }
        loader = loader
            .profile_name(name)
            .credentials_provider(credentials.build())
            .region(RegionProviderChain::first_try(region.build()).or_default_provider());
    }
    loader
}
