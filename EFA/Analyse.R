

library(Momocs)

hearts %T>%                    # A toy dataset
  stack() %>%                  # Take a family picture of raw outlines
  fgProcrustes() %>%           # Full generalized Procrustes alignment
  coo_slide(ldk = 2) %T>%      # Redefine a robust 1st point between the cheeks
  stack() %>%                  # Another picture of aligned outlines
  efourier(6, norm=FALSE) %>%  # Elliptical Fourier Transforms
  PCA() %T>%                   # Principal Component Analysis
  plot_PCA(~aut) %>%           # A PC1:2 plot
  LDA(~aut) %>%                # Linear Discriminant Analysis
  plot_CV()      


h <- hearts


coo <- bot[1]
coo
