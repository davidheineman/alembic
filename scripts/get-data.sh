#!/usr/bin/env bash
# Downloads CIFAR-10 (binary version) into public/data as two flat files of
# 3073-byte records: [label, 1024 R, 1024 G, 1024 B].
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p public/data
tmp=$(mktemp -d)
curl -L -o "$tmp/cifar.tgz" https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar xzf "$tmp/cifar.tgz" -C "$tmp"
cat "$tmp"/cifar-10-batches-bin/data_batch_{1,2,3,4,5}.bin > public/data/cifar10_train.bin
cp "$tmp/cifar-10-batches-bin/test_batch.bin" public/data/cifar10_test.bin
rm -rf "$tmp"
ls -la public/data
