data {
    int<lower=0> ngames;
    int<lower=1, upper=20> nteams;
    array[ngames] int hometeam; // home team index
    array[ngames] int awayteam; // away team index
    array[ngames] int<lower=0> y1;       // goals scored by home team
    array[ngames] int<lower=0> y2;
}
parameters {
    real mu_att;
    real mu_def;
    real home;
    real<lower=0> tau_att;         // precision of attacking abilities
    real<lower=0> tau_def;         // precision of defensive abilities
    vector[nteams] att_star;       // unscaled attacking abilities
    vector[nteams] def_star;       // unscaled defensive abilities
}
transformed parameters {
    vector[nteams] att;            // mean-centered attacking abilities
    vector[nteams] def;            // mean-centered defensive abilities
    vector[ngames] log_theta1;     // log scoring rate for home team
    vector[ngames] log_theta2;     // log scoring rate for away team

    att = att_star - mean(att_star);
    def = def_star - mean(def_star);

    for (g in 1:ngames){
        // Average Scoring intensities (accounting for mixing components)
        log_theta1[g] = home + att[hometeam[g]] + def[awayteam[g]];
        log_theta2[g] = att[awayteam[g]] + def[hometeam[g]];
    }

}
model {
  // LIKELIHOOD AND RANDOM EFFECT MODEL FOR THE SCORING PROPENSITY
  for (g in 1:ngames) {
    // Observed number of goals scored by each team
    y1[g] ~ poisson_log(log_theta1[g]);
    y2[g] ~ poisson_log(log_theta2[g]);

  }

  // prior on the home effect
  home ~ normal(0, sqrt(10));  // std. dev. instead of precision (like in jags)

  // priors on the random effects
  mu_att ~ normal(0, sqrt(10));
  mu_def ~ normal(0, sqrt(10));
  tau_att ~ gamma(.01,.01);
  tau_def ~ gamma(.01,.01);

  // Trick to code the sum-to-zero constraint
  att_star ~ normal(mu_att, tau_att);
  def_star ~ normal(mu_def, tau_def);

}
generated quantities {
  array[ngames, 2] int ynew;

  for (g in 1:ngames) {
    ynew[g, 1] = poisson_log_rng(log_theta1[g]);
    ynew[g, 2] = poisson_log_rng(log_theta2[g]);
  }
}
