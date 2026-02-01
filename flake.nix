{
  description = "guth - Fast CPU-optimized text-to-speech CLI";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "guth";
          version = "0.1.0";
          src = ./.;
          cargoLock.lockFile = ./Cargo.lock;
          nativeBuildInputs = [ pkgs.cmake pkgs.pkg-config ];
          meta = with pkgs.lib; {
            description = "Fast CPU-optimized text-to-speech CLI using flow matching";
            homepage = "https://github.com/kyutai-labs/pocket-tts";
            license = with licenses; [ mit asl20 ];
            maintainers = [ ];
          };
        };

        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            cargo
            rustc
            rust-analyzer
            clippy
            rustfmt
            cmake
            pkg-config
          ];
        };
      });
}
