# Databricks notebook source
# MAGIC %md
# MAGIC ### DATA INGESTION

# COMMAND ----------

# MAGIC %md
# MAGIC - Deployed Cluster
# MAGIC - Stored Data
# MAGIC - Downloaded data from 
# MAGIC [https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data]()
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### DATA EXPLORATION

# COMMAND ----------

from pyspark.sql import SparkSession
spark=SparkSession.builder.appName("spark_eceommerce_data").getOrCreate()

# COMMAND ----------

customers_df=spark.read.format("csv").option("header",'true').option("inferSchema","true").load("/FileStore/tables/olist_customers_dataset.csv")

location_df=spark.read.format("csv").option("header",'true').option("inferSchema","true").load("/FileStore/tables/olist_geolocation_dataset.csv")

order_items_df=spark.read.format("csv").option("header",'true').option("inferSchema","true").load("/FileStore/tables/olist_order_items_dataset.csv")

# COMMAND ----------

payments_df=spark.read.format("csv").option("header",'true').option("inferSchema","true").load("/FileStore/tables/olist_order_payments_dataset.csv")

reviews_df=spark.read.format("csv").option("header",'true').option("inferSchema","true").load("/FileStore/tables/olist_order_reviews_dataset.csv")

orders=spark.read.format("csv").option("header",'true').option("inferSchema","true").load("/FileStore/tables/olist_orders_dataset.csv")

products=spark.read.format("csv").option("header",'true').option("inferSchema","true").load("/FileStore/tables/olist_products_dataset.csv")

sellers=spark.read.format("csv").option("header",'true').option("inferSchema","true").load("/FileStore/tables/olist_sellers_dataset.csv")

translation=spark.read.format("csv").option("header",'true').option("inferSchema","true").load("/FileStore/tables/product_category_name_translation.csv")

# COMMAND ----------

for df in [
    customers_df, location_df, order_items_df, payments_df, reviews_df,
    orders, products, sellers, translation
]:
    df.printSchema()
    display(df.limit(5))

# COMMAND ----------

print(f'Customers count: {customers_df.count()}')
print(f'Orders count: {orders.count()}')

#No Data Lekage as the count is similar to the dataset count

# COMMAND ----------

#Null Values Count
from pyspark.sql.functions import *
customers_df.select([count(when(col(c).isNull(),1)).alias(c) for c in customers_df.columns]).display()
orders.select([count(when(col(c).isNull(),1)).alias(c) for c in orders.columns]).display()

# COMMAND ----------

#Duplicate Values
customers_df.groupBy('customer_id').count().filter(col("count")>lit(1)).show()

#that is no duplicate values

# COMMAND ----------

#Customer Distrubution By State

customers_df.groupBy('customer_state').count().orderBy(col('count').desc()).show(8)

# COMMAND ----------

#ORder Distrubution BY status
orders.groupBy('order_status').count().orderBy(col('count').desc()).show(8)

# COMMAND ----------

#Distribution of pyaments

payments_df.groupBy('payment_type').count().orderBy(col('count').desc()).show()

# COMMAND ----------

#Top selling products
top_products=order_items_df.groupBy('product_id').agg(sum('price').alias('Top Selling Product'))
top_products.orderBy(col('Top Selling Product').desc()).show(5)

# COMMAND ----------

#Average Delivery Time Analysis
orders.withColumn("deliver_time_from_purchase",datediff(col('order_delivered_customer_date'),col('order_purchase_timestamp'))).orderBy('deliver_time_from_purchase',ascending=False).display()


# COMMAND ----------

# MAGIC %md
# MAGIC ### DATA CLEANING AND TRANSFORMATION

# COMMAND ----------

def missing_values(df, df_name):
    print(f'Missing values in {df_name} dataframe')
    df.select([count(when(col(c).isNull(), 1)).alias(c) for c in df.columns]).show()

missing_values(customers_df, 'customers')
missing_values(orders,'ORders')

# COMMAND ----------

orders_df_cleaned=orders.na.drop(subset=['order_id','customer_id','order_status'])
orders.count()

# COMMAND ----------


orders_df_cleaned.count()

# COMMAND ----------

orders_df_cleaned=orders.fillna({'order_delivered_customer_date':'9999-12-30'})
orders_df_cleaned.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### IMPUTE MISSING VALUES

# COMMAND ----------

payments_df.show(10)

# COMMAND ----------

#There is no null value , so assigning a null value
payments_df_null=payments_df.withColumn("payment_value",when((col('payment_value')!=99.33) & (col('payment_value')!=24.39),col('payment_value')).otherwise(lit(None)))

payments_df_null.show(5)


# COMMAND ----------

#Now we have 2 null values
from pyspark.ml.feature import Imputer
imputer=Imputer()
imputer.setInputCols(['payment_value'])
imputer.setOutputCols(['payment_value_imputer'])
imputer.setStrategy('median')

imputer_model = imputer.fit(payments_df_null)
payments_df_imputed = imputer_model.transform(payments_df_null)
payments_df_imputed.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Standardizing the format

# COMMAND ----------

def print_Schema(df,df_name):
  print(f'Printing Schema for: {df_name}')
  df.printSchema()

print_Schema(orders,'Orders')


# COMMAND ----------

print_Schema(customers_df,'Customers')

# COMMAND ----------

orders_df_cleaned=orders_df_cleaned.withColumn("order_purchase_timestamp",to_date(col('order_purchase_timestamp')))
orders_df_cleaned.printSchema()

# COMMAND ----------

payments_df_imputed.show(5)

# COMMAND ----------

payments_df_imputed=payments_df_imputed.withColumn('payment_type',when(col('payment_type')=='boleto',lit('something')).otherwise(col('payment_type')))
payments_df_imputed.show()

# COMMAND ----------

customers_df.printSchema()

# COMMAND ----------

#Converting customer_zip_code_prefix to string
customers_df=customers_df.withColumn('customer_zip_code_prefix',col('customer_zip_code_prefix').cast('String'))
customers_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Remove Duplicate Records

# COMMAND ----------

customers_df_cleaned=customers_df.dropDuplicates(['customer_id'])
customers_df_cleaned.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### JOIN

# COMMAND ----------

order_with_details=orders.join(order_items_df,'order_id','left')\
    .join(payments_df_imputed,'order_id','left')\
    .join(customers_df,'customer_id','left')
order_with_details.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Engineering

# COMMAND ----------

#find total order value
order_with_details.groupBy('order_id').agg(sum('payment_value').alias('Total ORder by customer'))\
                                              .orderBy(col('Total ORder by customer').desc()).show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Advanced Transformation

# COMMAND ----------

quantiles=order_items_df.approxQuantile('price',[0.01,0.99],0.0)
low_cutoff,high_cutoff=quantiles[0],quantiles[1]

low_cutoff,high_cutoff

# COMMAND ----------

order_items_df.select('price').summary().show()

# COMMAND ----------

#Remove outliers
order_items_df_cleaned=order_items_df.filter((col('price')>=low_cutoff) & (col('price')<=high_cutoff))
order_items_df_cleaned.select('price').summary().show()

# COMMAND ----------

#STORING FINAL DATA

order_with_details.write.mode('overwrite').parquet('/FileStore/tables/cleaned_date.parquet')

# COMMAND ----------

# MAGIC %md
# MAGIC ### DATA INTEGRATION

# COMMAND ----------

orders.cache()

# COMMAND ----------

order_items_joined_df=orders.join(order_items_df,'order_id','inner')
order_items_products_df=order_items_joined_df.join(products,'product_id','inner')

# COMMAND ----------

order_items_products_sellers_df=order_items_products_df.join(sellers,'seller_id','inner')

# COMMAND ----------

full_orders_df=order_items_products_sellers_df.join(customers_df,'customer_id','inner')

# COMMAND ----------

full_orders_df=full_orders_df.join(location_df,full_orders_df.customer_zip_code_prefix==location_df.geolocation_zip_code_prefix,'left')

# COMMAND ----------

full_orders_df=full_orders_df.join(reviews_df,'order_id','left')

# COMMAND ----------

full_orders_df=full_orders_df.join(payments_df,'order_id','left')

# COMMAND ----------

full_orders_df.cache()

# COMMAND ----------

full_orders_df.display()

# COMMAND ----------

#Total Revenue Per Seller

seller_revenue_df=full_orders_df.groupBy('seller_id').agg(sum('price').alias('Total Revenue')).show()

# COMMAND ----------

#Total Orders per customer
total_orders=full_orders_df.groupby('customer_id').agg(count('*').alias("Total Order")).show()

# COMMAND ----------

#average Review score per seller
avg_review_score=full_orders_df.groupBy('seller_id').agg(avg('review_score').alias('Average Review Score')).show()

# COMMAND ----------

#Most Sold products (Top 10)
most_sold=full_orders_df.groupBy('product_id').agg(count('*').alias("Top 10 Sold")).orderBy(col('Top 10 Sold').desc()).show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optimized Joins For Data Integration
# MAGIC

# COMMAND ----------

order_items_products_sellers_df=order_items_products_df.join(broadcast(sellers),'seller_id','inner')
full_orders_df=full_orders_df.join(broadcast(reviews_df),'order_id','left')
full_orders_df.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Window Function

# COMMAND ----------

from pyspark.sql.window import Window
window_spec=Window.partitionBy('seller_id').orderBy(desc('price'))

# COMMAND ----------

#Rank top seller products per seller
top_sellers_df=full_orders_df.withColumn('rank',rank().over(window_spec)).filter(col('rank')>=5).show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Advance Aggregation and Enrichment

# COMMAND ----------

#Total Revenue and Average Order Value per customer
customers_spending_df=full_orders_df.groupBy('customer_id').agg(sum('price').alias('Total Spent'),avg('price').alias("Average Order Value")).orderBy(desc('Total Spent')).show()

# COMMAND ----------

full_orders_df.printSchema()

# COMMAND ----------

#Monthly Revenue and order count
full_orders_df=full_orders_df.withColumn('order_purchase_month',month('order_purchase_timestamp'))
full_orders_df.groupBy('order_purchase_month').agg(count('*').alias("Order Count")).orderBy(desc('Order Count')).show()

# COMMAND ----------

#Customer Retention Analysis (First and LAst Order)
customer_retention_df=full_orders_df.groupBy('customer_id').agg(first('order_purchase_timestamp').alias("First ORder"),last('order_purchase_timestamp').alias("Last ORder"),count('order_id').alias("Total ORders")).orderBy('Total ORders').show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### ENRICHMENT

# COMMAND ----------

full_orders_df = full_orders_df.withColumn(
    'is_delivered',
    when(col('order_status') == 'delivered', lit(1)).otherwise(lit(0))
).withColumn(
    'is_cancelled',
    when(col('order_status') == 'cancelled', lit(1)).otherwise(lit(0))
)

full_orders_df.select('order_status','is_delivered','is_cancelled').show()

# COMMAND ----------

#Order REvenue Calculation

full_orders_df=full_orders_df.withColumn('REvenue',col('price')+col('freight_value'))
full_orders_df.select('price','freight_value','REvenue').show()

# COMMAND ----------

#Customer Segmentation based on spending

customers_spending_df = customers_spending_df.withColumn(
    'Customer Segmentation',
    when(col('Average Order Value') >= 1200, 'High Value')
    .when(col('Average Order Value') >= 800, 'Medium Value')  # Assuming 800-1199 is Medium
    .otherwise('Low Value')
)


# COMMAND ----------

