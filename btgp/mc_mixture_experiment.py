"""
Here for reference; works in colab
Just need to import functions from the other files and get the data
"""

n_gps = 20
opt_steps = 5000
time_limit = 60*5 # stop training any given gp after this many seconds
update_order_level = 2 # 0 is no changes to the bit order; 3 is the most
performance_weighting = 100 # maybe sqrt(n data points) is a good anchor?

gp_pop = GPMixture(train_x.to(device=dev, dtype=torch.float64),
                      train_y.to(device=dev, dtype=torch.float64),
                      test_x.to(device=dev, dtype=torch.float64),
                      lambd=1e-6)
for _ in range(n_gps):
    gp = ONGP(train_x.to(device=dev, dtype=torch.float64),
          train_y.to(device=dev, dtype=torch.float64),
          test_x.to(device=dev, dtype=torch.float64), lambd=1e-6)
    gp.random_bit_order()
    gp.process()
    gp.update_weights_slsqp(max_iter=opt_steps, update_order_level=update_order_level, time_limit=time_limit)
    gp_pop.add(gp)
gp_pop.process_all()
# gp_pop.cull_bottom(quantile=0.25)
print("Train nlls:", gp_pop.get_nlls())
print("Test nlls:", gp_pop.get_test_nlls(test_y_cuda))
print("Mixture test nll:", gp_pop.gp_mixture_predict(test_y_cuda, weight_func=lambda nlls: torch.exp(-performance_weighting*nlls)))
