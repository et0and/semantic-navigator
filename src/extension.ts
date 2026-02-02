import * as vscode from 'vscode';

class FacetsTreeDataProvider implements vscode.TreeDataProvider<vscode.TreeItem> {
	getTreeItem(element: vscode.TreeItem): vscode.TreeItem {
		return element;
	}

	getChildren(element?: vscode.TreeItem): Thenable<vscode.TreeItem[]> {
        return Promise.resolve([
			new vscode.TreeItem('Facet 1'),
			new vscode.TreeItem('Facet 2')
		]);
	}
}

class FocusTreeDataProvider implements vscode.TreeDataProvider<vscode.TreeItem> {
	getTreeItem(element: vscode.TreeItem): vscode.TreeItem {
		return element;
	}

	getChildren(element?: vscode.TreeItem): Thenable<vscode.TreeItem[]> {
		return Promise.resolve([]);
	}
}
export function activate(context: vscode.ExtensionContext) {
	vscode.window.createTreeView('facet-navigator.views.explorer.facets', {
		treeDataProvider: new FacetsTreeDataProvider()
	});

	vscode.window.createTreeView('facet-navigator.views.explorer.focus', {
		treeDataProvider: new FocusTreeDataProvider()
	});

	const disposable = vscode.commands.registerCommand('facet-navigator.doSomething', () => {});

	context.subscriptions.push(disposable);
}

export function deactivate() {}