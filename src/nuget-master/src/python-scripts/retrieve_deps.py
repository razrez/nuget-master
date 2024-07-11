#!pip install PyGithub
import sys
import re
from github import Github, GithubException 
from github import Auth


def get_packages_from(repository: str):

    # Get the repository object
    repo = g.get_repo(repository)

    # Get all branches in the repository
    branches = repo.get_branches()

    # Get the default branch (usually the root branch)
    default_branch = next((branch for branch in branches if branch.name == repo.default_branch), None)

    # Get the root tree of the repository
    tree = repo.get_git_tree(sha=default_branch.commit.sha, recursive=True)

    # Filter out only the.csproj files
    csproj_files = []
    for item in tree.tree:
        if item.type == "blob" and item.path.endswith(".csproj"):
            csproj_files.append(item)

    # Extract PackageReference from each.csproj file
    package_references = set()
    for file in csproj_files:
        file_content = repo.get_contents(file.path).decoded_content.decode("utf-8")
        pattern = r'PackageReference Include="([^"]+)"'
        matches = re.findall(pattern, file_content)
        package_references.update(matches)

    return package_references
    

auth = Auth.Token("<token>") # using an access token
g = Github(auth=auth)

while True:
    try:
        repoName = sys.stdin.readline().strip()
        recommendedDepsString = get_packages_from(repoName)
        if (len(recommendedDepsString) == 0): 
            print(f"error: {repoName} have no references to third-party NuGet packages")
        else:
            print(",".join(recommendedDepsString))
        
    except GithubException as e:
        if e.status == 404:
            print(f"error: Repository not found")
        else:
            print(f"error: {e.message}")
    
    except Exception as ex: print(f"error : {ex}")