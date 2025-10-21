{
  description = "A basic multiplatform flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    # System types to support.
    supportedSystems = ["x86_64-linux" "x86_64-darwin" "aarch64-linux" "aarch64-darwin"];

    # Helper function to generate an attrset '{ x86_64-linux = f "x86_64-linux"; ... }'.
    forAllSystems = nixpkgs.lib.genAttrs supportedSystems;

    # Nixpkgs instantiated for supported system types.
    nixpkgsFor = forAllSystems (system: import nixpkgs {inherit system;});
  in {
    devShells = forAllSystems (system: let
      pkgs = nixpkgsFor.${system};
      python = pkgs.python312.withPackages (p: let
        pmdarima = p.pmdarima.overrideAttrs (old: {
          doCheck = false;
          doInstallCheck = false;
        });
      in [
        p.pandas
        p.jupyterlab
        p.jupyterlab-execute-time
        p.jupyterlab-widgets
        p.jupyterlab-lsp
        p.numpy
        p.matplotlib
        p.statsmodels
        p.datetime
        p.openpyxl
        p.seaborn
        p.scipy
        p.prophet
        p.sklearn-compat
        p.keras
        p.tabulate
        p.geopandas
        pmdarima
        p.rasterio
        p.folium
        p.nltk
        p.ftfy
        p.unidecode
        p.wordcloud
        p.transformers
        p.joblib
        p.torch
        p.python-lsp-server
        p.spacy
        p.spacy-models.es_core_news_sm
        p.spacy-lookups-data
        p.spacy-loggers
        p.spacy-alignments
        p.spacy-transformers
        p.networkx
        p.sentencepiece
      ]);
    in {
      default = pkgs.mkShell {
        packages = [
          python
          pkgs.black
        ];
      };
    });
  };
}
