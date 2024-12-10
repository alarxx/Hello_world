---
creation_time: 2024-12-10 04:18
parents:
  - "[[Software Packaging and Deployment System]]"
---
---

**Зачем нужна Anaconda?**

И Anaconda и [[Package Installer for Python (PIP)|PIP]] являются являются менеджерами среды и пакетов.

**Conda** устанавливает предварительно скомпилированные бинарные пакеты. 
С **[[Python - Virtual Environment|venv]]** некоторые библиотеки могут требовать компиляции или внешних C-библиотек, что отностельно долго и может быть проблемой на разных OS.

**Conda** управляет версией python, python-библиотеками и главное отличие - ==системными зависимостями==, как версия CUDA, например. 

**Anaconda** идет с множеством предустановленных библиотек:
NumPy, Pandas, Matplotlib/Seaborn, Scikit-learn, TensorFlow/PyTorch.
**Miniconda**: `conda install`.

---

**Miniconda**
https://d2l.ai/chapter_installation/index.html

**Installation**
https://docs.anaconda.com/miniconda/install/#miniconda-manual-shell-init-linux

Normal + Quick:
```sh
mkdir ~/miniconda3
cd ~/miniconda3
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > miniconda.sh
# bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
bash ./miniconda.sh
rm ./miniconda.sh
```

Там потом спросят хотите ли вы, чтобы `conda` всегда автоматически инициализировалась в терминале.
Если ответил no -> Manual Shell Initialization:
```sh
source ~/miniconda3/bin/activate
conda init --all # initialize on all available shells
```

Re-open terminal or run:
```sh
source ~/.bashrc
```

Теперь по умолчанию (base) environment, чтобы убрать run:
```sh
conda deactivate
```
or:
```sh
conda config --set auto_activate_base false
```
and to return:
```sh
conda activate base
```

---

https://chatgpt.com/c/67577567-0fb4-800d-a61f-14fb406ab095

**Environments**
https://docs.anaconda.com/working-with-conda/environments/

Creating (transaction):
```sh
conda create -n <ENV_NAME> python=<VERSION> <PACKAGE>=<VERSION>
```

```sh
conda create -n myenv python=3.11 beautifulsoup4 docutils jinja2=3.1.4 wheel
```

```sh
conda info --envs
conda activate <ENV_NAME>

conda deactivate
```

**Sharing an environment**

Export env config:
```sh
conda env export > environment.yml
```

>This file handles both the environment’s pip packages and conda packages.

Creating from .yml:
```sh
conda env create -f environment.yml
```


==Кажется, <ENV_NAME> должно быть уникальным для каждого проекта.==

Странно, можно скачивать в окружении conda, как с помощью **conda**, так и с помощью **pip**, но не рекомендуется скачивать с pip, только если бинарников библиотеки нет в conda репозиториях.

Как сменить версию python:
```sh
conda uninstall python # кажется это удаляет все зависимости
conda install python=3.9
```

```sh
conda install anaconda::jupyter
```
- defaults
- conda-forge
- anaconda - поддерживаемая [[Anaconda Inc.]]