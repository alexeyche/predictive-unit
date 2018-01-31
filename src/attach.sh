#!/usr/bin/env bash

set -ex

docker exec -it $(docker ps | sed '1d' | awk '{print $1}') /bin/bash
