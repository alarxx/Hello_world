---
creation_time: 2024-07-22 19:21
parents:
  - "[[Python]]"
  - "[[Debian]]"
---

---

Links:
- [webhosting.uk/](https://www.webhosting.uk.com/kb/how-to-install-python-on-debian-12/);
- [digitalocean/](https://www.digitalocean.com/community/tutorials/how-to-install-python-3-and-set-up-a-programming-environment-on-debian-11)

```sh
sudo apt install python3
sudo apt install python3-pip
#sudo apt install python3.8
python3 --version
```

# Virtual Environment

Try (but should be error):
```sh
pip3 install numpy  
```

> [!Error] Virtual Environment
```sh
error: externally-managed-environment  
  
× This environment is externally managed  
╰─> To install Python packages system-wide, try apt install  
   python3-xyz, where xyz is the package you are trying to  
   install.  
      
   If you wish to install a non-Debian-packaged Python package,  
   create a virtual environment using python3 -m venv path/to/venv.  
   Then use path/to/venv/bin/python and path/to/venv/bin/pip. Make  
   sure you have python3-full installed.  
      
   If you wish to install a non-Debian packaged Python application,  
   it may be easiest to use pipx install xyz, which will manage a  
   virtual environment for you. Make sure you have pipx installed.  
      
   See /usr/share/doc/python3.11/README.venv for more information.  
  
note: If you believe this is a mistake, please contact your Python installation or OS distribution provider.  
You can override this, at the risk of breaking your Python installation or OS, by passing --break-system-pack  
ages.  
hint: See PEP 668 for the detailed specification.
```

```sh
sudo apt install -y python3-venv
```

```sh
mkdir environments
cd environments
```

Create environment:
```sh
python3 -m venv my_env
```
	-m - module-name, finds sys.path and runs corresponding .py file 

```sh
ls my_env
```

Use this environment:
```sh
source my_env/bin/activate
```

>[!Note] Note
>Within the virtual environment, you can use the command `python` instead of `python3`, and `pip` instead of `pip3` if you would prefer. If you use Python 3 on your machine outside of an environment, you will need to use the `python3` and `pip3` commands exclusively.

---

# Workflow

Install Python and [[Package Installer for Python (PIP)|PIP]]:
```sh
sudo apt install python3
sudo apt install python3-pip
#sudo apt install python3.8
python3 --version
```

Install Virtual Environment Package:
```sh
sudo apt install -y python3-venv
```

Now can enter project directory:
```sh
cd path/to/project
```

Create environment:
```sh
python3 -m venv .venv
```
	-m - module-name, finds sys.path and runs corresponding .py file 

Use this environment:
```sh
source .venv/bin/activate
```

Add this environment to .gitignore:
```sh
echo ".venv" >> .gitignore
```

To recreate environment:
```sh
pip freeze > requirements.txt
```

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

Полный workflow от установки git и python, до слияния веток: 
[[psychic-pancake]].