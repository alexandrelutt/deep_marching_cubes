python scripts/build_extension_smoothness.py install
cd dist
unzip smoothness_extension-0.0.0-py3.10-linux-x86_64.egg
cd ..
mv dist smoothness_extension

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