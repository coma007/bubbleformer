![Header Image](https://github.com/coma007/bubbleformer/blob/main/docs/newspaper-header.png)
# News Articles Recommendation System - _bubbleformer_

Project for Petnica Seminar in Machine Learning (PSI:ML) 9.

### Authors & Mentors  
This project was collaboratively developed by:  
- [Tanja Miković, ETF Belgrade](https://github.com/wknjk)  
- [Milica Sladaković, FTN Novi Sad](https://github.com/coma007)  
  
We extend our gratitude to our mentors, whose guidance and expertise were instrumental in the success of this project:
- [Kosta Grujčić, Microsoft](https://rs.linkedin.com/in/kosta-grujcic)  
- [Stefan Mojsilović, Everseen](https://rs.linkedin.com/in/stefan-mojsilovic)
  


## Table of Contents

- [Objective](#objective)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Metrics](#metrics)
- [Results and Evaluation](#results-and-evaluation)
- [Examples](#examples)
- [Conclusion](#conclusion)
- [References](#references)
- [About PSI:ML](#about-psiml)

## Objective

The primary aims of this project are as follows:

1. **User-specific Article Recommendation:** Develop a network that predicts whether a news article is relevant to a specific user. The goal is to tailor recommendations to individual users based on their preferences and interactions (such as clicks). This involves analyzing the user's past behavior to understand their preferences and using this information to generate personalized article recommendations. The objective is to enhance the user experience by delivering articles that are more likely to align with their interests.

2. **Reconstruction of NRMS Results:** Replicate and implement the findings from the paper ["Neural News Recommendation with Multi-Head Self-Attention (NRMS)"](https://wuch15.github.io/paper/EMNLP2019-NRMS.pdf) [1].  This objective involves implementing the methodology outlined in the paper and reproducing the results presented in it. By replicating the results, the project can verify the effectiveness of the proposed approach and gain insights into the performance of the model under different conditions or datasets.


## Dataset

The project utilizes the MIND (Microsoft News Dataset), which comprises the following components:

- **News Articles:** The dataset includes 50,000 news articles, each with information such as the headline, abstract, category, URL, and more.
- **Users:** There are 50,000 users within the dataset.
- **Click Logs:** The dataset contains 150,000 click logs, documenting users' click history, recommended articles, and associated click information, including timestamps.

The data collection process took place over a span of 6 weeks during October and November 2019, with the following breakdown:

- The first 4 weeks involved creating the history of selected articles.
- The 5th week's first 6 days were used for the training set.
- The last day of the 5th week was allocated for the validation set.
- The 6th week was dedicated to the test set.

This diverse dataset allows the project to explore and develop personalized article recommendation systems.

For more information, refer to the [MIND Dataset documentation](https://msnews.github.io/assets/doc/ACL2020_MIND.pdf). [2]


## Architecture

The solution architecture is designed to create an effective and personalized news article recommendation system. It consists of three core components, each contributing to the system's ability to understand user preferences and deliver relevant content.

### 1. News Encoder

The News Encoder component focuses on processing news article titles to extract meaningful representations for further analysis and comparison. It follows these steps:
- **Word2Vec Encoding:** The news article titles are encoded using the Word2Vec technique. This transforms words into high-dimensional vectors, capturing semantic relationships between words.
- **Multi-Head Self Attention:** The encoded word vectors undergo Multi-Head Self Attention, allowing the model to weigh the importance of different words within the title based on their contextual relevance. This step enables the model to recognize intricate patterns and contexts in the title.
- **Additive Self Attention:** To obtain a comprehensive title representation, Additive Self Attention is applied. This final representation encapsulates the key features of the title that contribute to its overall meaning.

### 2. User Encoder

The User Encoder component focuses on capturing the preferences and interests of individual users by analyzing the news articles they have interacted with. The process involves the following stages:
- **Aggregation of News Articles:** All news articles read by a user are aggregated and channeled through Multi-Head Self Attention. This step allows the model to understand the relationships between articles in a user's reading history.
- **Additive Self Attention for User Bubble:** The output of the previous step goes through Additive Self Attention, generating a user representation that resembles a "bubble" of related news articles. This representation encapsulates the user's interests and preferences based on their interaction history.
  
### 3. Click Predictor

The Click Predictor component determines the likelihood of a user clicking on a given news article. This prediction is crucial for delivering recommendations that align with user preferences. The process involves:
- **Dot Product Operation:** The user representation and the news article representation are combined using a dot product operation. This operation measures the similarity between the user's interests and the content of the news article.


![Architecture Image](https://github.com/coma007/bubbleformer/blob/main/docs/architecture.png)

The overall architecture synergizes these three components to create a recommendation system that takes into account both the content of news articles and the historical interactions of users. By leveraging advanced attention mechanisms and representations, the system provides users with personalized news article suggestions that cater to their individual preferences.


## Metrics

The performance of the recommendation system is rigorously evaluated using the following metrics, each offering unique insights into its effectiveness:

1. **Area Under Curve (AUC):**
   Area Under Curve quantifies the model's ability to distinguish between positive and negative instances, offering a comprehensive assessment of the ranking quality. In the context of the recommendation system, AUC measures how well the system can differentiate between articles that the user would engage with and those they would not. A higher AUC score indicates better discrimination and implies that the system effectively ranks articles based on user preferences.

2. **Mean Reciprocal Rank (MRR):**
   Mean Reciprocal Rank focuses on the position of the first relevant item in the recommendations list. It emphasizes the system's capability to place the most relevant articles at the top. In our project, MRR evaluates the efficiency of the recommendation system in terms of placing articles that the user would find engaging within the initial positions of the recommendation list. A higher MRR score signifies that the system excels at presenting top-relevant articles early on.

3. **Normalized Discounted Cumulative Gain (nDCG@k):**
   Normalized Discounted Cumulative Gain takes into account both the relevance of recommended items and their positions in the list. The nDCG@k metric, computed at a specified cutoff (k) in the recommendation list, assesses the quality of recommendations by considering the diminishing returns of relevance as we move down the list. In our project, nDCG@k reflects how well the system suggests relevant articles within the first k positions. A higher nDCG@k score indicates a better balance between relevance and ranking.

These metrics collectively offer a comprehensive evaluation of the recommendation system's performance. By utilizing them, we gain insights into different aspects of the system's effectiveness, including ranking quality, relevance of recommendations, and the order in which articles are presented. This multi-faceted assessment ensures that our recommendation system is not only accurate in prediction but also adept at delivering highly relevant articles to users early in their recommendation feed.


## Results and Evaluation

### Loss Function

The loss function serves as a guiding principle during the training of our recommendation system. It facilitates the model in adapting and improving its predictions based on user interactions. To achieve this, we adopt a loss function that draws inspiration from prior research (Huang et al., 2013) and utilizes negative sampling techniques.

Negative sampling involves selecting both clicked and non-clicked articles to establish a balanced learning process. For each news article a user engages with (considered a positive sample), we randomly choose K other articles from the same set that were not clicked by the user (considered negative samples). These samples collectively form a classification task, with the objective of predicting the probability of user clicks.

The loss function's purpose is to minimize the discrepancy between predicted and actual click probabilities. By doing so, the model learns to assign higher probabilities to articles a user would likely click on and lower probabilities to those they would not. This adjustment process enables the recommendation system to become more precise in its suggestions over time.

This loss function adaptation has been found effective in refining our recommendation system, aligning it with user preferences and enhancing the relevance of suggestions.

![Loss Image](https://github.com/coma007/bubbleformer/blob/main/docs/loss.png)

### Metric Performance

The performance of the recommendation system is assessed using metrics, which provide insights into its effectiveness. The results for individual metrics are available in the image below:

![Metric Results](https://github.com/coma007/bubbleformer/blob/main/docs/results.png)

Results from original paper are mentioned bellow our results, in parenthesis. Notably, it is worth mentioning that several metrics achieved results that are comparable to or even surpass the original paper's reported values. This underlines the system's robustness and potential for delivering enhanced recommendations.


## Examples

In the example bellow, the table consists of two columns: the left column for the random selected user's history and the right column for the recommendations for that user. Each row corresponds to one news article.

| User History                                        | Recommendations                                     |
|-----------------------------------------------------|-----------------------------------------------------|
| Former **Trump** advisor John Bolton…               | McConnell: Impeachment measure …                    |
| **Diplomats**: **Iran** briefly held IAEA …         | **Trump** says jihadist leader died …               |
| **Trump’s** advisers gave him opti …                | Ocasio-Cortez grills Facebook’s Zuc ..              |
| Elton John has not seen “Bohemian..                 | Elijah Cummings farewell: **Obama**, …              |
| Anonymous **Trump** official writing …              | Pelosi holds off vote to **Trump** imp …            |
| Man accused in probe of Guiliani …                  | House **Democrat** criticizes Mississippi …         |
| CNN Poll: **Biden’s** lead in Democrat …            | On impeachment strategy, **Pelosi** …               |
| Celebs turning 60 in 2019                           | Kevin Durant: Draymond Green Fig …                  |
| **Iran** downs drone over southern …                | What **Trump** has said about Ukraine               |
| **Iran** nuclear crisis escalates                   | Jordan: **Republicans** Will Subpoena…              |

## Conclusion

The successful replication of results from the original paper underscores the validity and robustness of our approach. Implementing the entire network from scratch, however, is a time-consuming endeavor that demands substantial resources.

Further Research Directions:

1. **Positional Encoder Enhancement:** Exploring the incorporation of positional encoders for both news headlines and user interaction history could potentially enhance the model's understanding of sequential information, leading to improved recommendations.

2. **Context-Aware News Sampling:** Moving beyond random sampling, the system could benefit from context-aware news sampling techniques. This approach would ensure that sampled articles are relevant to both user preferences and the current context.

3. **Exploring Advanced Architectures:** Diving deeper into other architectures that have demonstrated superior results might yield insights for further enhancing the performance of our recommendation system.

As we move forward, these research avenues offer promising opportunities for refining and expanding the capabilities of our recommendation system.


## References

[1]  Wu et al. (2019).
   [Neural News Recommendation with Multi-Head Self-Attention](https://wuch15.github.io/paper/EMNLP2019-NRMS.pdf)
   [Original Paper]

[2]  Wu et al. (2020)
   [MIND: A Large-scale Dataset for News Recommendation](https://msnews.github.io/assets/doc/ACL2020_MIND.pdf)
   [Dataset Paper]

[3]  Reference code: [News Recommendation GitHub Repository](https://github.com/yusanshi/news-recommendation)
   [GitHub Repository]

These references played a crucial role in shaping the concepts, methodologies, and implementations within this project.


## About PSI:ML

This project was undertaken as part of the [Petnica Seminar in Machine Learning (PSI:ML) 9](https://psiml.pfe.rs/). PSI:ML is an intensive educational program organized by [PFE](https://pfe.rs/) that immerses participants in a deep exploration of machine learning concepts, techniques, and applications.

The Petnica Seminar in Machine Learning (PSIML) serves as a platform for students to delve into the intricate world of machine learning, fostering an environment of collaborative learning and experimentation. The results of this project reflect the dedication and expertise cultivated through the PSIML experience.

We are grateful to PSI:ML & PFE Crew for providing the foundation and support for this project, enabling us to explore and contribute to the field of machine learning.  💕

![Footer Image](https://github.com/coma007/bubbleformer/blob/main/docs/newspaper-footer.png)
