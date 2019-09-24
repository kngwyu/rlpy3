#!/bin/bash
set -ex

cd /io

for PYBIN in /opt/python/cp{35,36,37}*/bin; do
    export PYTHON_SYS_EXECUTABLE="$PYBIN/python"
    "${PYBIN}/pip" install -U setuptools wheel==0.31.1 cython
    "${PYBIN}/python" setup.py bdist_wheel
done

for whl in dist/*.whl; do
    auditwheel repair "$whl" -w dist/
done
