# ML production deployment

Production follows `main`. Pushes run forecasting, VRP and Lambda-contract tests, then build Linux amd64 images and exercise container startup. Image-validation smoke tests run a real CLIP inference with a fixture image and a read-only filesystem. PRs build/test without deployment credentials. Successful production builds push commit-tagged images to the existing ECR repositories and update Lambda using the resolved digest. Failed Lambda smoke tests restore the previous image and fail the workflow.

| Source | Function | ECR repository |
| --- | --- | --- |
| `hotspot_forecasting` | `ecopin-temp-hotspot-forecasting-aka-spatial-analysis` | `hotspot-forecasting` |
| `vrp_service` | `ecopin-vrp-service` | `ecopin-vrp-service` |
| `image_validation/inference_service_exp32.py` | `clip-validator` | `clip-validation-lambda` |

The image-validator container packages the tracked Exp 3.2 RegMLP checkpoint and downloads the frozen CLIP extractor at build time. Its native Lambda handler preserves the existing API Gateway `POST /clip-validator` JSON `{ "image": "<base64>" }` contract. `GET` on the same API route reports health. No API routes or public-access policies are created by this workflow.

## One-time prerequisites

1. Run the reviewed AWS bootstrap from the backend repository; it creates role `ecopin-ml-github-deploy` with trust limited to `klacom/ecopin_ml_models` on `main`. No AWS credentials need to be saved in GitHub.
2. Preserve existing Lambda environment settings while adding current `SUPABASE_URL` and `SUPABASE_SECRET_KEY` to forecasting. Use a protected local file/AWS Console, not workflow variables or committed files.
3. The existing forecasting and VRP functions are configured with just 128 MB and a three-second timeout. Before the first deployment, set forecasting to 1024 MB/120 seconds and VRP to 1024 MB/90 seconds. Keep image validation at 3008 MB and raise its timeout to 120 seconds for cold starts. These settings can increase per-invocation cost; there is no provisioned concurrency.
4. Preserve existing API Gateway integrations and Lambda URLs. Backend `USE_AWS_CLASSIFIER=true` uses the existing classifier gateway URL. Set `USE_AWS_SPATIAL_FORECAST=true` plus `AWS_SPATIAL_FORECAST_SERVICE_URL` to the existing forecasting Function URL. Keep `VRP_SERVICE_URL` pointed at its existing `/solve_vrp` endpoint.
5. Merge the latest forecasting/VRP branch and the current `main` without discarding either branch's changes, then add this deployment workflow. The feature branch is three commits ahead and one behind `main` at preparation time.

## Rollback

In ECR, select a known-good image digest for the affected service, then run:

```bash
python scripts/deploy_lambda.py forecasting <repository-uri>@sha256:<digest>
```

Replace `forecasting` with `vrp` or `image-validation` as needed. The smoke test expects the current service contract; older model versions may need manual AWS rollback. Config/environment rollback is separate from image rollback. Deployments of the three services are independent, so one service's failure does not roll back successful deployments of the other two.

Verification includes unit tests, real Linux container startup and CLIP inference. The forecasting deployment smoke test calls the real database-backed forecast endpoint, not just health. The Python Supabase client is pinned to the locally verified version supporting modern secret keys. Check GitHub Actions for the currently deployed commit and live verification outcome; local Docker Desktop is unavailable and builds run on GitHub-hosted Linux runners.
