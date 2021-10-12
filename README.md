# Drift-Detection-in-Streaming-Data

Drift detection is an important concept in the analysis of streaming data. Drift can be defined as any prominent outlier in the streaming data window which needs to be identified as a 'drift' or not belonging to any previously identified strain of classes. Drift detection has been addressed by many algorithms, the most popular ones among them being Early Drift Detection Method (EDDM) among others.

In this project, I propose a novel way to detect drifts in streaming data: one found to be 10% more accurate than EDDM. The project makes use of a customized K-Means clustering mechanism, wherein the Cumulative Sum of Squares (CSS) is modified into CUCSS (Continuously Updating Cumulative Sum of Sqaures) across the streaming data windows for continuously adding known drifts to existing classes on their adequate occurrence, while watching out for and predicting new drifts in future streaming windows.

### Technology used: Pandas, Recursive Feature Extraction (RFE), Principal Component Analysis (PCA), Logistic Regression, Auto encoder, Onehot encoder, Matplotlib
