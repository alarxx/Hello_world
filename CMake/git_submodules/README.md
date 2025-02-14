
# Description

https://stackoverflow.com/questions/31769820/differences-between-git-submodule-and-subtree

**`git submodule`** - это когда мы ссылаемся на внешний репозиторий, используем его, но он не интегрирован в наш исходный код, он не включается в коммиты нашего репозитория, у нас как-бы 2 отдельных репозитория, репозиторий внутри репозитория.

Другой подход - `git subtree` - команда включает исходный код в наш репозиторий, у нас вес самого проекта увеличивается. Мы присваиваем себе библиотеку и можем ее свободно менять.

**2025-02-01**

Я не использовал `git subtree`, но теоретически с ней ты присваешь себе код библиотеки полностью, это окей, если библиотека под MIT, BSD, Apache, Unlicense, но появляются трудности, если библиотека развивается и ты хочешь merge-ить себе свежие обновления библиотеки.

На сколько я сейчас понимаю, если ты используешь Open Source (OSS) библиотеки и не против commit-ить, то лучше всего использовать **`git submodule`**, потому что в таком случае мы можем легко и за-merge-ить к себе обновления библиотеки и так же легко можем commit-тить обратно в библиотеку.

Если в OSS библиотека под MPL и ты меняешь содержимое файлов, то тебе нужно коммитить их обратно, и `git submodule` будет полезен.
- Mozilla Public License (MPL)

Тем более, если ты используешь LGPL библиотеку, то `submodule` - отличный вариант, потому что ты не можешь менять ничего локально, все изменения должны быть за-commit-чены обратно. LGPL нельзя link-овать статически, но можно link-овать динамически, не включая в исполняемый файл, и лицензировать свой проект по своему.
Говоря про LGPL стоит отметить про copyleft, что FOSS библиотеки под (A)GPL ты не можешь использовать не лицензировав свой софт так же под (A)GPL в не зависимости от типа link-овки (static или dynamic).
- Static and Dynamic Linking
- General Public License (GPL)
- Lesser General Public License (LGPL)
- Affero General Public License (AGPL)
- Server Side Public License (SSPL)

---

# Brief Summary

Практичны буквально 2 команды:

Adding:
```sh
git submodule add <url>
```
Update and pull:
```sh
git submodule update --flag
```
- `--init`
- `--remote` - fetching new commits
- `--recursive`

---

Initialize project using `git clone <url>` or:
```sh
git init
git remote add <origin> <url>
git fetch
git branch -r
git checkout <branch>
```

Adding submodule:
```sh
git submodule add <url>
git submodule update --init --resursive
```

Specify branch:
```sh
git config -f .gitmodules submodule.<dir/submodule>.branch <branch>

git submodule update --remote <submodule>
```

Fetching all updates (new commits):
```sh
git submodule update --remote
```

Cloning:
```sh
git clone <url>
git submodule update --init --resursive
```

Pulling updates:
```sh
git pull
git submodule update --init --recursive
```

---

# Starting

References:
- **git docs**: https://git-scm.com/book/en/v2/Git-Tools-Submodules
- delete submodule: https://stackoverflow.com/questions/1260748/how-do-i-remove-a-submodule
- specify branches: https://stackoverflow.com/questions/1777854/how-can-i-specify-a-branch-tag-when-adding-a-git-submodule

**Main repository**
1) Создать Main remote repository в GitHub
2) Подключиться к Main repository
	- `git clone <url>`
	- `git remote add`
```sh
git init
# remote GitHub repository
git remote add <origin> <url>
git fetch
# check remote branches
git branch -r
git branch <branch_name>
```


**Adding submodule** in our main repository:
```sh
git submodule [--quiet] add [-b <branch>] [-f|--force] [--name <name>] [--reference <repository>] [--] <repository> [<path>]
```
```sh
# Лучше всего не добавлять <dir/RepoName>, а просто из директории вызывать submodule add
mkdir external
cd external

git submodule add <url/RepoName> <dir/RepoName>
git status
# new file: .gitmodules
# new file: <dir>/RepoName
```
Заметь, что Git видит `<dir>/RepoName` как файл, хотя это директория.

_.gitmodules_:
```c
[submodule "RepoName"]
	path = <dir/RepoName>
	url = <url/RepoName>
```

```sh
# General info
git diff --cached
# basically, info about the file containing the commit hash
git diff --cached RepoName
# submodules info in .gitmodule
git diff --cached --submodule
```
**Импортированный submodule `<dir>/RepoName` - это файл или директория?**
`<dir>/RepoName` - это mode 160000 файл в котором записана ссылка HEAD commit hash. Фича Git. Эта информация хранится в `.git/index` и в commit-ах.
- mode 100644 - обычный файл
- mode 160000 - submodule file в котором записана ссылка HEAD commit hash.


---

**Delete submodule**

Как удалить submodule:
```sh
git rm -f <submodule>
# Also, you may remove local cache:
rm -rf .git/modules/<submodule>
```
and then commit.

**Зачем `rm -rf .git/modules/<submodule>`?**
После `git rm`, локально остается submodule. Вообще, это видимо сделали для оптимизации запросов, если ты случайно удалил, например (over-engineering).
Use-case, если захочешь обратно добавить этот submodule приходится писать `--force`:
```sh
git submodule add --force <url/RepoName> <dir/RepoName>
```
`--force` нужен потому что git локально оставляет submodule.
Поэтому после удаления еще нужно удалить локально `.git/modules/<submodule>` - остатки.

---

**Private URL**
Note
Люди будут pull-ить из public URL-а - `<url/RepoName>`.
Если ты хочешь push-ить в другое место, то можно локально указать другой URL:
```sh
# url перезаписать локально во время разработки
git config submodule.RepoName.url PRIVATE_URL
```

---

# Cloning a Project with Submodules

**Method 1. Simple Cloning**

Cloning superproject:
```sh
git clone <url>

```
После этого superproject склонируется, но submodule директории будут пустыми.

Чтобы за-fetch-ить submodule-и:
```sh
# init local configuration, mode 160000
git submodule init
# git submodule deinit <submodule>

# fetch submodules with appropriate commit
git submodule update
```
Можно объединить эти две команды `submodule init` и `submodule update` в одну:
```sh
git submodule update --init
```

---

**Method 2. Clone nested submodules recursively**

Либо, все можно было записать в одной команде `clone`:
```sh
git clone --recurse-submodules <url>
```
Команда должна склонировать проект и подтянуть все nested submodules recursively.

---

==И следующий метод самый лучший.==

Если уже склонировал, то можно использовать:
```sh
git submodule update --init --resursive
```

-> [[Git Submodule - Circular Dependency]]

---

# Working

#### fetching submodule updates (new commits)
Подтянуть новейшие commit-ы для библиотеки.

Проще всего, когда вы не меняете код, а просто хотите получать обновления submodule-я.

Cloning
```sh
git clone <url>
git submodule update --init --recursive
```

Merging
```sh
cd ./<submodule>

git fetch
git merge <branch>

cd ../
```
==Но, лучше следующее==
Если не хочешь заходить в каждый submodule и merge-ить обновления:
```sh
git submodule update --remote <submodule>
```
По-умолчанию подтянет данные из ветки на которую до этого указывал HEAD commit submodule-я.

---

**Differences**
```sh
# Покажет, что изменился HEAD commit
git diff
# Покажет list of commits и какие файлы в submodule были изменены
git diff --submodule
```
Если не хочешь постоянно добавлять flag `--submodule`, то можно установить его по умолчанию:
```sh
git config --global diff.submodule log
git diff
# Now same as with --submodule flaf
```

---

**Specify branches**

1) `.gitmodules` - for everyone
2) `.git/config` - local

```sh
git config -f .gitmodules submodule.<dir/submodule>.branch <branch>

git submodule update --remote
```
Если пропустить `-f .gitmodules`, то branch изменится локально только для тебя.

---

submodule summary?
```sh
git config status.submodulesummary 1
git status
git diff --submodule

git commit ...
git log -p --submodule
```

Тут прикольные flag-и для `git log`:
`-p` - открывает файлы
remote`--submodule` - внутри submodule directory

---

#### pull updates

Теперь посмотрим со стороны collaborator-а.
Если мы обновили версию библиотеки, то collaborator-у недостаточно просто сделать `pull`, после нужно update-нуть submodule-и:
```sh
git pull
git submodule update --init --recursive
```
Если не update-нуть submodule-и collaborator может попытаться обратно откатить.

Объединенная команда с помощью flag-а:
```sh
git pull --recurse-submodules
```
Можно указать это flag по умолчанию (я не уверен в команде):
```sh
git config submodule.recurse true
```

**sync url**
Команда update может не сработать, если URL of remote repository of submodule changed, и не получится найти commit соответственно (кажется URL сохраняется локально?). Такое может произойти, если submodule поменял hosting platform (GitHub -> GitLab).
```sh
git submodule sync --recursive
git submodule udpate --init --recursive
```

# Questions

**1. git submodule add, можно ли указать ветку?**
```sh
git submodule add -b <branch> <url>
```
**И можно ли сменить ветку сабмодуля уже после того как мы added.**
```sh
git config -f .gitmodules submodule.<dir/submodule>.branch <branch>

git submodule update --remote --recursive
```


**2. Когда мы обновляем информация об этом где-то сохраняется?**
Да, команда обновит ссылку на новейший commit
```sh
git submodule update --remote
```
- `--remote` - подтягивает новейшие commit-ы.
**Типа если коллаборатор на своем компе попытается забилдить мой проект у него какая версия сабмодуля  установится?**
Установится по указанию commit-а
```sh
git pull
git submodule update --init --recursive
```

**3. Почему где-то пишут что submodules плохие и лучше использовать subtree?**
Трудно пользоваться.

-> [[Git Submodule - Circular Dependency]]


Может ли команда вызвать [[Circular Dependency]]?
```sh
git submodule update --init --recursive
```

Команда будет fetch-ить все вложенные submodule-и.

==Но, Circular Dependency невозможен.==

---

Допустим есть две библиотеки:
- A
- B

A.0
A.1 -> B.0
B.0
B.1 -> A.1
A.2 -> B.1

A.2 -> B.1 -> A.1 -> B.0
Заметим, что тут нет никаких циклов.

---

**Более подробно**

A.0
Ссылаем A на новейшую B:
```sh
git submodule add <url>
git commit -m "A.1 -> B.0"
git push
```
При этом A увеличивается на commit.
A.1 -> B.0

B.0
Ссылаем B на новейшую A (A.1):
```sh
git submodule add <url>
git commit -m "B.1 -> A.1"
git push
```
При этом B увеличивается на commit, но новейший A не будет на него ссылаться, а будет на прошлый commit.
B.1 -> A.1

A.1
Теперь из уже существующего A подтянем обновления:
```sh
git submodule update --remote
git commit -m "A.2 -> B.1"
git push
```
A увеличивается на commit.
A.2 -> B.1

A.2:
```sh
git clone <url>
git submodule update --init --recursive
```

A.2 -> B.1 -> A.1 -> B.0

---

Я подумал, что `--recursive` может начать выполняться бесконечно, если будет Circular Dependency, но мы же не просто ссылаемся на библиотеку, а на определенный commit.
Да, плохо если библиотеки ссылаются друг на друга, но Circular Dependency не будет, не будет выполняться вечно.

---

**Local Circular Dependency**

А что если выполнить?
```sh
git submodule update --remote --recursive
```
Эта команда подтянет новейшие commit-ы для всех библиотек.

Допустим мы вызвали команду из A.2.
Тогда локально можно сделать Circular Dependency.

A.2 -> B.1 -> A.1 -> B.0

A.2 -> B.1 -> A.2 -> B.1 -> A.2

Да, теперь получается что библиотеки ссылаются на новейшие версии друг друга, но толку нет. Попробуй за-commit-ить, ничего не изменилось ведь, A.2 ссылался на B.1, как было так и осталось.


И еще, теперь если вызвать:
```sh
git submodule update --init --recursive
```
Кажется, у нас откатится до
A.2 -> B.1 -> A.1 -> B.0
