{ pkgs }: {
    deps = [
      pkgs.vim
pkgs.nodejs-16_x
      pkgs.nodePackages.yarn
      pkgs.replitPackages.jest
      pkgs.nodePackages.typescript-language-server
    ];
}