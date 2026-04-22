class ConfigTP:
    """Configuration for MF-Former on the WeatherBench dataset."""

    # basic settings
    model_name = "mf_former"
    ini_params_mode = "xavier"
    threshold = 10

    # dataset settings
    dataset = "WeatherBench"
    dataset_root = "./data"

    # sequence settings
    in_seq_len = 6
    out_seq_len = 6
    seq_len = in_seq_len + out_seq_len
    seq_interval = None

    input_gap = 1
    input_length = 6
    output_length = 6
    pred_shift = 6

    # device settings
    use_gpu = True
    num_workers = 8
    device_ids = [0, 1]

    # optimization settings
    train_batch_size = 8
    valid_batch_size = 8
    test_batch_size = 8
    train_max_epochs = 20
    learning_rate = 1e-4
    optim_betas = (0.5, 0.999)
    scheduler_gamma = 0.5

    # logging and checkpoint
    model_train_fre = 1
    loss_log_iters = 100
    model_save_fre = 1
    log_dir = "./logdir_weatherbench"
    save_path = "./saved_models"


config_tp = ConfigTP()