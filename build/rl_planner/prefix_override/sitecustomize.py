import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/media/bach/bach/nav_expl/CogniPlan_humble/install/rl_planner'
