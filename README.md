# Price elasticity estimation using regression and double ML

## Introduction
According to [Economic Research Service of U.S Department of Agriculture](https://www.ers.usda.gov/data-products/chart-gallery/gallery/chart-detail/?chartId=58322), apple and orange juice are America's favorite juice. Almost 50% of all juice consumed by Americans is ornage juice. The goal of this report is analyzing supermarket scanner data to investigate this claim. If Americans indeed love orange juice, the price should be more inelastic. i.e. price increase should not stop Americans from buying orange juices!

Question: how does the previous week's price affect this week's quantities sold?

## Data Source 
Alan L. Montgomery (1997), "Creating Micro-Marketing Pricing Strategies Using Supermarket Scanner Data," Marketing Science 16(4) 315–337.

## Variables
- store: store number
- brand: brand indicator
- week: week number
- logmove: log of the number of units sold
- price#: price of brand #
- deal: in-store coupon activity
- feature: whether there was feature advertisement for the brand at the store # on the week.
- profit: profit of the juice
- AGE60: percentage of the population that is aged 60 or older
- EDUC: percentage of the population that has a college degree
- ETHNIC: percent of the population that is black or Hispanic
- INCOME: median income
- HHLARGE: percentage of households with 5 or more persons
- WORKWOM: percentage of women with full-time jobs
- HVAL150: percentage of households worth more than $150,000
- SSTRDIST: distance to the nearest warehouse store
- SSTRVOL: ratio of sales of this store to the nearest warehouse store
- CPDIST5: average distance in miles to the nearest 5 supermarkets
- CPWVOL5: ratio of sales of this store to the average of the nearest five stores

## Result
The price elasticity from linear regression is -0.0655. If price increase by 1%, then the quantity decreases by 0.06%. The optimal markup is 16.7%, which is lower than the expectation of 30-40% markup.
The price elasticity from double ML is -0.023085. If price increase by 1%, then the quantity decreases by 0.02%. The optimal markup is 50%, which is slightly higher than the expectation.

## Further Questions
- Multiple aritcles claim Tropicana as the best selling orange juice of US. I want to examine the elasticity of tropicana to measure the market power and markup of the brand. In doing so, inter-temporal subtitution and cross elasticities would need to be controlled for.
- Examine the effect of being featured on quantity sold. This might be tricky to examine the causual relationship since it is likely that price are lower when the brand is featured. The assumption of Double ML is that E[μ|t, X]=0. In other words, price setters do not act on anything that allows them to predict sales that is not in the data. Retailers features a product because they anticipate sales to increase when there is price drop, then this ananlysis does not work.
- Which type of store has higher profit? The ones with more, or those with less variety of brands/sizes?
- What demographics of people prefer which brand?
- Which brands are most competitive (i.e. largest magnitude of cross-price elasticity)
