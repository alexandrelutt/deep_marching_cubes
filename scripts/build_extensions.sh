python scripts/build_extension_curvature.py install
cd dist
unzip curvature_extension-0.0.0-py3.10-linux-x86_64.egg
cd ..
mv dist curvature_extension

python scripts/build_extension_occto.py install
cd dist
unzip occtopology_extension-0.0.0-py3.10-linux-x86_64.egg
cd ..
mv dist occtopology_extension

python scripts/build_extension_dist.py install
cd dist
unzip distance_extension-0.0.0-py3.10-linux-x86_64.egg
cd ..
mv dist distance_extension