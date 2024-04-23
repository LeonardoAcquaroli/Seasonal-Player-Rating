class ClusterFeaturesWeightsConstants:
    # Drop not useful and sparse columns
    GAME_TEAM_IDS = [
    'game_id',
    'team_id'
    ]

    DBINFO=['competition',
    'team',
    'full_name',
    # 'cluster_label',
    'gametime',
    'player_id',
    'competition_id',
    'season',
    'home_team_id',
    'away_team_id',
    'opponent_team_id',
    'match_day',
    'side',
    'stat_type',
    'created_at',
    'updated_at']

    METRICS_TO_DROP = [
    'att_bx_centre',
    'att_obx_centre',
    'att_bx_right',
    'att_bx_left',
    'att_goal_high_centre',
    'att_goal_high_left',
    'att_goal_high_right',
    'att_goal_low_centre',
    'att_goal_low_left',
    'att_goal_low_right',
    'att_hd_goal',
    'att_hd_miss',
    'att_hd_post',
    'att_hd_target',
    'att_hd_total',
    'att_ibox_blocked',
    'att_ibox_goal',
    'att_ibox_miss',
    'att_ibox_post',
    'att_ibox_target',
    'att_lf_goal',
    'att_lf_target',
    'att_lf_total',
    'att_miss_high',
    'att_miss_high_left',
    'att_miss_high_right',
    'att_miss_left',
    'att_miss_right',
    'att_obox_blocked',
    'att_obox_goal',
    'att_obox_miss',
    'att_obox_post',
    'att_obx_left',
    'att_obx_right',
    'att_obox_target',
    'att_obxd_left',
    'att_obxd_right',
    'att_obp_goal',
    'att_lg_centre',
    'att_lg_left',
    'att_lg_right',
    'att_one_on_one',
    'att_cmiss_high',
    'att_cmiss_high_right',
    'att_cmiss_high_left',
    'att_cmiss_left',
    'att_cmiss_right',
    'att_openplay',
    'att_pen_goal',
    'att_pen_miss',
    'att_pen_post',
    'att_pen_target',
    'att_post_high',
    'att_post_left',
    'att_post_right',
    'att_rf_goal',
    'att_rf_target',
    'att_rf_total',
    'att_sv_high_centre',
    'att_sv_high_left',
    'att_sv_high_right',
    'att_sv_low_centre',
    'att_sv_low_left',
    'att_sv_low_right',
    'attempts_conceded_ibox',
    'attempts_conceded_obox',
    'game_started',
    'winning_goal',
    'goals_set_pieces_faced',
    'goals_big_chance_faced',
    'opp_touches',
    'team_touches',
    'team_touches_attack',
    'team_touches_buildup',
    'team_touches_defensive',
    'player_touches',
    'sub_position',
    'end_minute',
    'start_minute',
    'second_yellow',
    'passes_left',
    'passes_right',
    'leftside_pass',
    'rightside_pass',
    'pen_goals_conceded'
    ]

    METRICS_GK = [
    'clean_sheet',
    'dive_catch',
    'dive_save',
    'gk_smother',
    'goal_kicks',
    'good_high_claim',
    'keeper_pick_up',
    'keeper_throws',
    'saved_ibox',
    'saved_obox',
    'saves',
    'six_second_violation',
    'stand_catch',
    'stand_save',
    'penalty_faced',
    'total_keeper_sweeper',
    'accurate_keeper_sweeper',
    'accurate_keeper_throws',
    'diving_save',
    'goals_prevented',
    'Goals_prevented_per_xGoT',
    'set_pieces_faced',
    'big_chance_faced',
    'accurate_goal_kicks',
    'keeper_goals',
    'punches',
    'penalty_save'
    ]

    # Goals are to consider or not?
    METRICS_GOALS = [
    'goals_conceded',
    'goals_conceded_ibox',
    'goals_conceded_obox',
    'goals_openplay',
    'goal_fastbreak'
    ]

    METRICS_ASSISTS = [
        # 'goal_assist',
        'goal_assist_setplay',
        'goal_assist_openplay',
        'goal_assist_intentional',
        'second_goal_assist',
        'assist_own_goal',
        'goal_assist_deadball'
    ]

    columns_to_remove = GAME_TEAM_IDS + DBINFO + METRICS_TO_DROP + METRICS_GK + METRICS_GOALS + METRICS_ASSISTS