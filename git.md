# Git

## 1. Creating Repository
- Initializing a repository
  ```
  git init
  ```
- Clone from a repository
  ```
  git clone ssh_addres_of_repo
  ```


## 2. Stage Area
- add files to stage phase
  
  copy the current version of files to stage. it means if we change them we need to add them again to the stage phase.
  1. A signle file:
  ```
  git add file.txt
  ```
  2. Multiple file:
  ```
  git add file.txt app.py index.html
  ```
  3. Every file in currect directory (recursively on subfolders):
  ```
  git add .
  ```
  4. Add every Python file
  ```
  git add *.py
  ```
- Unstage (Undo add operation)
  
  when we restore something, the current version of that file in local will stay with no change.
  ```
  git restore --staged [file][files][files with specific patter]
  ```
  example: `git restore --staged file1.txt main.py`
- Removing file
  1. Removes from working directory and staging area
    ```
    git rm file1.js 
    ```
  2. Removes from staging area only
    ```
    git rm --cached file1.js
    ```
- Renaming or moving files (Affects both in stage area and directory)
  ```
  git mv file.txt main.js
  ```
- Showing files in stage area
  ```
  git ls-files
  ```


## 3. Viewing the status
  1. Full Status
     ```
     git status
     ```
  2. Short Status
     ```
     git status -s
     ```


## 4. Committing

The staging area in Git holds data for the next commit, which is a snapshot of the project’s current state. After a commit, the staging area seems empty as `git status` shows no files ready for commit. This is because they’ve just been committed, making the stage identical to the last commit. So, `git status` shows nothing to commit. However, `git ls-files` will still list the files in the staging area.
- Commit the pre-staged files
  1. Commits with a one-line message
  ```
  git commit -m “Message”
  ```
  2. Opens the default editor to type a long message
  ```
  git commit
  ```
- skipping the stage area: directly commit without staging the changes. also `-m "Message"` can be used for short messages.
  ```
  git commit -a
  ```

## 5. Push, Pull

- Pull
- Push
  ```
  git push
  ```



## 6. Viewing the changes

  `git diff` is used for knowing differences between two versions of a file.
  - show unstaged changes

    stage area VS local
    ```
    git diff FILE_NAME
    ```
  - show staged changes

    last commit VS stage area
    ```
    git diff --staged FILE_NAME
    ```
    

## 7. Browsing History

  `HEAD` is used for latest commit. we can also use a `~NUMBER` after the `HEAD` to points previous commits refferencing by HEAD. like `HEAD~3` wich points to 3 commits before. another way is to point a commit with its unique ID.
  - Showing commit history: use `space` to go to next page and `q` to quit.
    1. Full history
      ```
      git log
      ```
    2. Summary
      ```
      git log --oneline
      ```
    3. List the commits from the oldest to the newest
      ```
      git log --reverse
      ```
  - All the files on the directory in a specific commit
    ```
    git ls-tree TARGET_COMMIT
    ```
    example: `git ls-tree HEAD`, `git ls-tree b01fe02`
  - Showing inside of an file: first we should know its ID.
    ```
    git show file_ID
    ```
    example: `git show 94954abda49de`


## 8. Restoring
- Discarding local changes
  1. Copies old file in stage area (index) to current file in working directory
     ```
     git restore [file][files][files with specific patter]
     ```
  2. Discard all local changes (except untracked files)
     ```
     git restore .
     ```
  3. Removes all untracked files
     
     Some files are new and not tracked by Git yet. So, when we run the previous command (restore this file), Git doesn’t know where to get the previous version of this file, because it doesn’t exist in our staging area nor the repository. Therefore, to remove all these new untracked files, we should run:
     ```
     git clean -fd
     ```
- Restore to a previous commit version

  Suppose we remove a file from the local and staging area and also commit that change. Now, we’re regretting our decision and want to restore it, but we don’t have it in our staging area. Therefore, we should use backups to go back to the previous version before this commit::
  ```
  git restore --source=TARGET_COMMIT FULL_PATH_TO_THE_FILE
  ```
  example: `git restore --source=HEAD~1 app.py`
    

## Other
- Git ignore
  1. add a folder in the directory to `.gitignore`
      ```
      echo foldername/ >> .gitignore
      ```
  2. all text files should be ignored
      ```
      echo "*.txt" >> .gitignore
      ```

## Setup
- Installing Git
  - Windows:
    1. Install `winget tool` if you don't already have it.
    2. In powershell:
       ```
       winget install --id Git.Git -e --source winget
       ```
  - Linux/Unix
    ```
    apt-get install git
    ```
- git version
```
git --version
```


## Configuration Settings
- different settings:
  1. `--system`: All users
  2. `--global`: All repositories of the current user
  3. `--local`: the current repository
- Typical configurations
```
git config --global user.name "FisrtName LastName"
git config --global user.email example@gg.com
git config --global core.editor "code --wait"
```
- Open our default editor to edit global config file
```
git config --global -e
```


## Visual Tools

1. Using VS Code with `git difftool` instead of `git diff`: you should first add these lines to your `.config` file (don't try to add them with terminal)
   ```
   [diff]
     tool = vscode
   [difftool "vscode"]
     cmd = "code --wait --diff $LOCAL $REMOTE"
   ``` 
