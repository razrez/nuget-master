// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as vscode from 'vscode';
import { commands, window, TreeItemCollapsibleState } from 'vscode';
import { spawn } from 'child_process';
import {Options, PythonShell} from 'python-shell';
import * as path from 'path';

let repoRecommenderShell: PythonShell;
let retrieveDepsShell: PythonShell;
let description: any = "";
let repositoryForFetch: any = "";
let nugetTreeDataProvider: NugetTreeDataProvider;
const pathToScripts = path.join(__dirname, '../src', 'python-scripts');
let recommenderOptions: Options = {
	mode: 'text',
	pythonOptions: ['-u'], // get print results in real-time
	scriptPath: pathToScripts,
	args: [description],
	timeout: 2147483646
};

let depsRetrieverOptions: Options = {
	mode: 'text',
	pythonOptions: ['-u'], // get print results in real-time
	scriptPath: pathToScripts,
	args: [repositoryForFetch],
	timeout: 2147483646
};

class TreeNode extends vscode.TreeItem {
	constructor(
		public readonly label: string,
		public readonly elementType: 'repository' | 'dependency',
		public readonly collapsibleState: vscode.TreeItemCollapsibleState,
	) {
		super(label, collapsibleState);

		if (this.elementType === 'dependency'){
			this.checkboxState = vscode.TreeItemCheckboxState.Unchecked;
		}
	}
}

export class NugetTreeDataProvider implements vscode.TreeDataProvider<TreeNode> {
    private _onDidChangeTreeData: vscode.EventEmitter<TreeNode | undefined | null | void | TreeNode[] > = new vscode.EventEmitter<TreeNode | undefined | null | void | TreeNode[]>();
	
	private _selectedDependencies: TreeNode[] = [];
	private _repositories: TreeNode[] = [];
	private _dependencies: TreeNode[] = [];

	
    getTreeItem(element: TreeNode): TreeNode {
		// console.log('getTreeItem:', element);
      	return element;
    }
  
    getChildren(element?: TreeNode): Thenable<TreeNode[]> {
		if (!element) {
			return this.getRepositories();
		} 
		
		else if (element.elementType === 'repository') {
			repositoryForFetch = element.label.toString()
			return this.getPackagesOf(repositoryForFetch);
		}

		return Promise.resolve([]);
    }

	// resets TreeView and invokes getChildren()
	newQuery(): void {
		// console.log('newQuery()');
		this._dependencies = [];
		this._repositories = [];
		this._selectedDependencies = [];
	}

    private async getRepositories(): Promise<TreeNode[]> {
		description = await window.showInputBox({
			prompt: 'Enter the textual description of the project',
			placeHolder: 'e.g. "jwt auth with dotnet6"'
		});

		if (!description) return [];

		repoRecommenderShell.send(description);
		repoRecommenderShell.end;

		return new Promise<TreeNode[]>((resolve) => {
			repoRecommenderShell.on('message', (message: string) => {
				if (message === "error"){
					resolve([])
					window.showErrorMessage(`Try again and use Latin letters `);
				}

				const recommendedRepositories = message.split(",");
				this._repositories = recommendedRepositories.map((repo) => new TreeNode(
					repo,
					'repository',
					TreeItemCollapsibleState.Collapsed
				));

				resolve(this._repositories);
			});
		});
    }
  
    private async getPackagesOf(repo: string): Promise<TreeNode[]> {

		retrieveDepsShell.send(repo);
		retrieveDepsShell.end;
		window.showInformationMessage(`Scrapping NuGet packages references in "${repo}"...`);

		return new Promise<TreeNode[]>((resolve) => {
			retrieveDepsShell.on('message', (message: string) => {
				if (message.startsWith("error")){
					resolve([])
					window.showErrorMessage(message);
				}

				const depsNames = message.split(",");
				const recommendedDependecies = depsNames.map((dependency) => {
					const newTreeNode = new TreeNode(
						dependency, 
						'dependency',
						TreeItemCollapsibleState.None,
					);
					
					// append new dependency to list of dependies (если нет с таким же названием)
					if (!this._dependencies.filter(d => d.label === dependency)) this._dependencies.push(newTreeNode);
					
					return newTreeNode
				});
				
				resolve(recommendedDependecies);
			});
		});
    }

	// selects/unselects TreeNode
	onDidChangeCheckboxState(checkboxChangeEvent: vscode.TreeCheckboxChangeEvent<TreeNode>): void {
		const dependencyTreeNode = checkboxChangeEvent.items[0][0];
		const dependencyTreeNodeCheckboxState = checkboxChangeEvent.items[0][1];

		if (dependencyTreeNodeCheckboxState === vscode.TreeItemCheckboxState.Checked){
			// console.log(`${dependencyTreeNode.label} is {Checked}`)
			this._selectedDependencies.push(dependencyTreeNode);
		}

		else {
			// console.log(`${dependencyTreeNode.label} is {Unchecked}`)
			const otherDeps = this._dependencies.filter((dependency) => 
				dependency.label !== dependencyTreeNode.label
			);

			this._selectedDependencies = otherDeps;
		}
	}

	async installDependencies() {
		// Get the list of selected dependencies
		const selectedDependencies = this._selectedDependencies.map((dependency) => dependency.label);
	  
		// Get the list of .csproj files in the workspace
		const csprojFiles = await searchCsprojFiles();

		if(csprojFiles){
			// Create a quick pick to select the project file to install dependencies
			const quickPick = window.createQuickPick();
			quickPick.items = csprojFiles.map((file) => ({ label: file }));
			quickPick.placeholder = 'Select a project to install dependencies';
			quickPick.show();
		  
			// Handle the selection of a project file
			quickPick.onDidChangeSelection((selection) => {
			  if (selection && selection[0]) {
				const projectFile = selection[0].label;
				this.installDependenciesToProject(projectFile, selectedDependencies);
			  }
			});
		} else {
			window.showInformationMessage(`".csproj" files wasn't found`);	
		}
	  
	}
	  
	installDependenciesToProject(projectFile: string, dependencies: string[]) {

		window.showInformationMessage("installing...");	
		// Create a child processes to execute the dotnet command
		dependencies.forEach((dependency) => {
			const childProcess = spawn('dotnet', ['add', 'package', dependency], {
				cwd: path.dirname(projectFile),
			});
		
			childProcess.stdout.on('data', (data) => {

				console.log(`stdout: ${data}`);
			});
		
			childProcess.stderr.on('data', (data) => {
				console.error(`stderr: ${data}`);
			});
		
			childProcess.on('close', (code) => {
				if (code === 0) {
					window.showInformationMessage(`${dependency} successfully installed to ${projectFile}`);
				} else {
					window.showErrorMessage(`Failed to install ${dependency} to ${projectFile}`);
				}
			});
		});
	}

}

async function searchCsprojFiles(): Promise<string[]> {
	const csprojFiles: string[] = [];
	await vscode.workspace.findFiles('**/*.csproj', '**/node_modules/**', 10).then((uris) => {
		uris.forEach((uri) => {
			csprojFiles.push(uri.fsPath); 
		});
	});

	return csprojFiles;
}

export function activate(context: vscode.ExtensionContext) {
	console.log(pathToScripts);
	nugetTreeDataProvider = new NugetTreeDataProvider();
	repoRecommenderShell = new PythonShell('repo_recommender.py', recommenderOptions);
	retrieveDepsShell = new PythonShell('retrieve_deps.py', depsRetrieverOptions);

	const recommendCommnad = commands.registerCommand('nuget-master.recommend', () => {
		nugetTreeDataProvider.newQuery();
		const nugetMasterTreeView = window.createTreeView('nugetMaster', {
			treeDataProvider: nugetTreeDataProvider,
			showCollapseAll: true,
		});
		nugetMasterTreeView.onDidChangeCheckboxState(e => nugetTreeDataProvider.onDidChangeCheckboxState(e));
	});
	
	const installDepsCommnad = commands.registerCommand('nuget-master.installDeps', async() =>
		 await nugetTreeDataProvider.installDependencies() 
	);
	
	context.subscriptions.push(recommendCommnad, installDepsCommnad);

}