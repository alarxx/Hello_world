
# CUDA Installation

## GPUs

**Verify You Have a CUDA-Capable GPU**
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#verify-you-have-a-cuda-capable-gpu
```sh
lspci | grep -i nvidia
```

**Your GPU Compute Capability:** https://developer.nvidia.com/cuda-gpus
Нужен NVIDIA GPU with minimum compute capability of 3.0.
Кажется, моя 1660 не поддерживает Tensor Core.

- https://forums.developer.nvidia.com/t/does-the-latest-gtx-1660-model-support-cuda
	- https://www.techpowerup.com/gpu-specs/geforce-gtx-1660.c3365

## CUDA Core vs. Tensor Core
- https://stackoverflow.com/questions/47335027/what-is-the-difference-between-cuda-vs-tensor-cores
- https://stackoverflow.com/questions/72932132/where-does-the-third-dimension-as-in-4x4x4-of-tensor-cores-come-from

- **CUDA**: 1 single precision multiply-accumulate in FP32 `a += b * c` per GPU clock
- **Tensor Core**: 1 4x4 matrix multiply-accumulate in mixed FP32-FP16 per GPU clock

https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/

---

## Download CUDA

https://developer.nvidia.com/cuda-downloads

Full script:
```sh
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda-repo-debian12-12-8-local_12.8.1-570.124.06-1_amd64.deb
sudo dpkg -i cuda-repo-debian12-12-8-local_12.8.1-570.124.06-1_amd64.deb
sudo cp /var/cuda-repo-debian12-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
```

#### My Interpretation
I don't use `sudo`, but `su`, and I use `apt` instead of `apt-get`.

1: (without changes)
```sh
wget https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda-repo-debian12-12-8-local_12.8.1-570.124.06-1_amd64.deb
```

`su`
```sh
# without it it sends error message
export PATH=$PATH:/usr/local/sbin:/usr/sbin:/sbin

dpkg -i ./cuda-repo-debian12-12-8-local_12.8.1-570.124.06-1_amd64.deb
```
or just use APT? IMO it is better:
```sh
apt install ./cuda-repo-debian12-12-8-local_12.8.1-570.124.06-1_amd64.deb
```

3:
```sh
apt update

apt install cuda-toolkit-12-8
```

Install drivers:
```sh
apt install nvidia-open
apt install cuda-drivers
```

Reboot and check
```sh
nvidia-smi
nvcc --version # won't work
```

---

## NVCC

[nvcc --version command says nvcc is not installed](https://askubuntu.com/questions/885610/nvcc-version-command-says-nvcc-is-not-installed)

Нужно добавить cuda исполяемые файлы и библиотеки в PATH вручную.

Add in `/home/.bashrc`:
```sh
# CUDA
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```

Re-open terminal or run:
```sh
source ~/.bashrc
```
Check again:
```sh
nvcc --version
```

---

[[Add Russian (Cyrillic) Locale in Terminal]]

---

## Remove

```sh
apt-get remove --purge '^cuda.*'
apt-get remove --purge '^nvidia.*'
apt-get autoremove -y
apt-get autoclean
```

```sh
sudo rm -rf /usr/local/cuda*
sudo rm -rf /var/cuda-repo-debian12-12-8-local
```

```sh
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt-get update
```

```sh
rm -f cuda-repo-debian12-12-8-local_12.8.1-570.124.06-1_amd64.deb
```

```sh
cd /usr/share/keyrings/
rm cuda-930170B2-keyring.gpg
```
