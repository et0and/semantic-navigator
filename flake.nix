{ outputs = { flake-utils, nixpkgs, self }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages."${system}";

      in
        { packages.default =
            let
              python = pkgs.python3.withPackages (python: [
                python.aiofiles
                python.aiohttp
                python.aiohttp-retry
                python.gitpython
                python.openai
                python.pinecone-client
                python.scikit-learn
                python.tiktoken
                python.xxhash
              ]);

            in
              pkgs.writeShellApplication {
                name = "facets";

                runtimeInputs = [ python ];

                text = ''python ${./facets.py} "$@"'';
              };
        }
    );
}
