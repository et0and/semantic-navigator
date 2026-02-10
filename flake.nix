{ inputs.pyproject-nix = {
    url = "github:pyproject-nix/pyproject.nix";

    inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { nixpkgs, pyproject-nix, ... }:
    let
      inherit (nixpkgs) lib;

      forAllSystems = lib.genAttrs lib.systems.flakeExposed;

      project = pyproject-nix.lib.project.loadPyproject {
        projectRoot = ./.;
      };

    in
      { devShells = forAllSystems (system:
          let
            pkgs = nixpkgs.legacyPackages.${system};

            python = pkgs.python3;

            renderer = project.renderers.withPackages { inherit python; };

            pythonEnv = python.withPackages renderer;

          in
            { default = pkgs.mkShell { packages = [ pythonEnv ]; }; }
        );

        packages = forAllSystems (system:
          let
            pkgs = nixpkgs.legacyPackages.${system};

            python = pkgs.python3;

            renderer = project.renderers.buildPythonPackage {
              inherit python;
            };

          in
            { default = python.pkgs.buildPythonPackage renderer; }
        );
      };
}
