Edge-AI enabled smishing and suspicious SMS detection: a cost-sensitive and lightweight deep learning architecture
Author: Huynh Cong Phap
Journal: Journal of Information and Telecommunication (2026)
DOI: https://doi.org/10.1080/24751839.2026.2713250

If this research is relevant to your work, please cite the published article:
Phap, H. C. (2026). Edge-AI enabled smishing and suspicious SMS detection: a cost-sensitive and lightweight deep learning architecture. Journal of Information and Telecommunication, 1–15. https://doi.org/10.1080/24751839.2026.2713250


Overview
Smishing and suspicious SMS detection pose two important challenges: asymmetric misclassification costs and limited computational resources on edge devices.
This research investigates a lightweight, cost-sensitive deep learning architecture combining Bidirectional Gated Recurrent Units (BiGRU), Masked Global Pooling, and Weighted Binary Cross-Entropy (WBCE).
The study examines predictive performance, cross-domain generalization, cost-sensitive decision thresholds, and deployment-oriented efficiency.
Key Contributions
1.	Cost-sensitive risk formulation: A mathematical interpretation of weighted binary cross-entropy through risk minimization under a reweighted probability measure.
2.	Lightweight neural architecture: BiGRU with Masked Global Pooling to exclude padding positions from aggregated sequence representations.
3.	Cross-domain evaluation: Experiments using UCI SMS Spam Collection and LSDST-2022 with a harmonized suspicious-SMS label space.
4.	Deployment efficiency: A compact model with low CPU inference latency in the reported benchmarking environment.
5.	Operational risk analysis: Investigation of decision thresholds under different false-negative and false-positive costs.
Selected Experimental Findings
Metric	Proposed model	DistilBERT reference
Serialized model footprint	5.21 MB	256.02 MB
CPU model-only latency	0.25 ms	40.78 ms
These figures are reported under the experimental profiling conditions described in the paper. DistilBERT is used as a footprint and latency reference, not as a jointly fine-tuned predictive-performance baseline.
Datasets
•	UCI SMS Spam Collection
•	LSDST-2022
Dataset access, preprocessing, binary label harmonization, and experimental splits should follow the published methodology.
Reproducibility
The research also includes a Google Colab implementation:
https://colab.research.google.com/drive/1f39iLfjNz9E7x7BlyUL37HIt67PJu7Ue
Additional verified source code, configuration files, experiment instructions, and reproducibility materials can be organized in this repository.
Research Applications
This work may be relevant to research on:
•	SMS phishing and smishing detection
•	Cost-sensitive classification
•	Imbalanced learning and asymmetric error costs
•	Lightweight neural networks for Edge-AI
•	Security-oriented natural language processing
•	Cross-domain robustness of short-text classifiers

Contact
Huynh Cong Phap
Vietnam–Korea University of Information and Communication Technology, The University of Danang, Vietnam
Email: hcphap@vku.udn.vn
ORCID: https://orcid.org/0009-0004-2548-1637
Publication
Published online: August 6, 2026.
Publisher: Taylor & Francis.
Full article: https://doi.org/10.1080/24751839.2026.2713250

