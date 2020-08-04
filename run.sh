#!/bin/sh

env PYTHONUNBUFFERED=true gunicorn \
    --workers 1 \
    web:app -b 0.0.0.0:5201
