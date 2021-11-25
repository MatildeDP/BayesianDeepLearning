import optuna

study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
