rm -r outputs
mkdir google_cloud
gcloud storage cp -r gs://npm3d google_cloud
mv google_cloud/npm3d outputs
rm -r google_cloud