# Disk Setup

Optional Ubuntu guide for mounting a secondary volume at `/data` before running large dataset jobs.

If the repository and generated datasets comfortably fit on the boot disk, you do not need this guide. The project itself does not require `/data`; output locations are controlled by run/config paths.

## 1. Identify the Volume

~~~bash
lsblk
~~~

Example:

~~~text
NAME   MAJ:MIN RM  SIZE RO TYPE MOUNTPOINT
vda    252:0    0   50G  0 disk
└─vda1 252:1    0   50G  0 part /
vdc    252:32   0 1280G  0 disk
~~~

A fresh data disk usually has no mount point.

## 2. Check Whether It Already Has a Filesystem

~~~bash
sudo file -s /dev/vdc
~~~

If the result is only `data`, it is unformatted. If a filesystem is reported, do **not** format it unless you intend to destroy the existing contents.

## 3. Format a Fresh Disk

> This destroys all existing data on the selected device.

~~~bash
sudo mkfs.ext4 /dev/vdc
~~~

Replace `/dev/vdc` with the actual data device.

## 4. Mount It

~~~bash
sudo mkdir -p /data
sudo mount /dev/vdc /data
sudo chown "$USER":"$USER" /data
~~~

Verify:

~~~bash
df -h /data
~~~

## 5. Persist Across Reboots

Find the UUID:

~~~bash
sudo blkid /dev/vdc
~~~

Add the UUID to `/etc/fstab`:

~~~text
UUID=<your-uuid>  /data  ext4  defaults  0  2
~~~

Validate the entry before rebooting:

~~~bash
sudo mount -a
~~~

No output indicates success.

## 6. Point Pretraining Output at the Volume

`configs/configure_synthetic.py` uses `${DATA_DIR}` in the generated pretraining configuration.

For example:

~~~bash
export DATA_DIR=/data/slm-synthetic-data/runs
make pretrain-smoke
~~~

SFT/DPO/distillation run roots are Make variables and can be overridden explicitly if desired.

## See Also

- [Command Reference](COMMANDS.md)
- [Generation Workflow](GENERATION_WORKFLOW.md)
