#!/bin/bash
set -ex

cd /io

for PYBIN in /opt/python/cp{36,37,38}*/bin; do
    export PYTHON_SYS_EXECUTABLE="$PYBIN/python"
    "${PYBIN}/pip" install -U setuptools cython
    "${PYBIN}/python" setup.py bdist_wheel
done

for whl in dist/*.whl; do
    auditwheel repair "$whl" -w dist/
done
