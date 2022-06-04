#! /usr/bin/bash
export PYTHONPATH=`pwd`:$PYTHONPATH

python -u tools/create_data.py rope3d \
		--root-path /mnt/data/Rope3D/ \
		--out-dir /mnt/data/Rope3D/ \
		--extra-tag rope3d \
		--with-plane
