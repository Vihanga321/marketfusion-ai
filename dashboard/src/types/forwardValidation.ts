export interface ForwardMetricSet {
  balanced_accuracy?: number | null;
  macro_f1?: number | null;
  log_loss?: number | null;
  brier?: number | null;
  cost_aware?: number | null;
}

export interface ForwardContractStatus {
  active_trading_days: number;
  matured_observations: number;
  directional_outcomes: number;
  wait_or_neutral_outcomes?: number;
  minimum_sample_ready: boolean;
  decision: string;
  failed_gates?: string[];
  context?: ForwardMetricSet;
  paired_baseline?: ForwardMetricSet;
  frozen_training_class_prior_log_loss?: number | null;
  balanced_accuracy_delta?: number | null;
  paired_log_loss_improvement?: number | null;
}

export interface ForwardValidationStatus {
  contract_version: string;
  asset_id: string;
  updated_at_utc?: string | null;
  decision: string;
  forward_start_utc?: string | null;
  contracts: Record<string, ForwardContractStatus>;
  historical_tuning_stopped?: boolean;
  models_frozen?: boolean;
  retraining_during_forward_window?: boolean;
  promotion_eligible?: boolean;
  model_promotion_performed?: boolean;
  production_integration?: boolean;
  automatic_execution?: string;
  runtime?: string;
  manual_confirmation?: string;
}
