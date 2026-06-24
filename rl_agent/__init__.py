from catchy_run_game._compat import install_legacy_module_alias

# Allow legacy checkpoints (pickled under the old "catchy_run." namespace) to load.
install_legacy_module_alias()
