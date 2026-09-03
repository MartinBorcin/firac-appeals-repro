# Anonymous Review Release

The tracked file content is prepared for anonymous review. Public dataset and
project URLs are represented by `DATASET_REPOSITORY_URL` until the camera-ready
release.

Before publishing a review mirror:

1. build `artifacts/MANIFEST.json` and run its verifier;
2. run the test, lint, credential, personal-path, and institution-domain checks
   listed in the root README;
3. publish a squashed snapshot or fresh anonymous repository so local commit
   author metadata and historical remotes are not transferred;
4. confirm that the anonymous paper URL, rather than a permanent repository URL,
   is used in the submission.

For the camera-ready release, replace `DATASET_REPOSITORY_URL` and restore the
public authorship and permanent project links in the citation metadata and
documentation.
