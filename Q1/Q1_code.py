from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, expr,regexp_extract, sum as spark_sum
import matplotlib.pyplot as plt

spark = SparkSession.builder \
    .master("local[2]") \
    .config("spark.local.dir","/mnt/parscratch/users/acp25ra") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")

logFile=spark.read.text("Data/NASA_access_log_Jul95").cache()
hosts = logFile.selectExpr("split(value, ' ')[0] as host")


#Section A
academicHosts = hosts.filter(hosts.host.contains(".ac.uk")).cache()
companyHosts = hosts.filter(hosts.host.contains(".co.uk")).cache()
governmentHosts = hosts.filter(hosts.host.contains(".gov.uk")).cache()


totalAcademic = academicHosts.count()
totalCompanies = companyHosts.count()
totalGovernments = governmentHosts.count()

uniqueAcademic = academicHosts.distinct().count()
uniqueCompanies = companyHosts.distinct().count()
uniqueGovernments = governmentHosts.distinct().count()


print("\n\nThere are %i total number requests from Academic institutions.\n" % (totalAcademic))
print("There are %i total number of requests from Companies.\n\n" % (totalCompanies))
print("There are %i total number of requests from Government departments.\n\n" % (totalGovernments))


print("\nThere are %i unique hostnames from Academic institutions.\n" % uniqueAcademic)
print("There are %i unique hostnames from Companies.\n" % uniqueCompanies)
print("There are %i unique hostnames from Government departments.\n" % uniqueGovernments)

# Section B

academicInstitutions = academicHosts.select(
    expr("concat_ws('.', slice(split(host, '\\\\.'), -3, 3)) as institution")
)


institutionCounts = academicInstitutions.groupBy("institution").count()
institutionCounts.orderBy(col("count").desc())

#Sheffield Total
sheffieldTotal = institutionCounts.filter(col("institution") == "shef.ac.uk").collect()[0]["count"]
numMoreThanSheffield = institutionCounts.filter(
    (col("institution") != "shef.ac.uk") & (col("count") > sheffieldTotal)
).count()


print("Total number of requests from the University of Sheffield domain: %i" % sheffieldTotal)
print("Number of other UK academic institutions with more requests than Sheffield: %i" % numMoreThanSheffield)

moreThanSheffieldDF = institutionCounts.filter(
    (col("institution") != "shef.ac.uk") & (col("count") > sheffieldTotal)
).orderBy(col("count").desc())
moreThanSheffieldDF.show()

#Save to csv
moreThanSheffieldDF.toPandas().to_csv("Q1_table1.csv", index=False)


#Section C
companyDomains = companyHosts.select(
    expr("regexp_extract(host, '([^.]+\\.co\\.uk)$', 1) as company")
).filter(col("company") != "")

companyCounts = companyDomains.groupBy("company").count()

top9Companies = companyCounts.orderBy(col("count").desc()).limit(9)

print("Top 9 most active UK companies by total number of requests:")
top9Companies.show(truncate=False)

totalCompanyRequests = companyCounts.agg(spark_sum("count").alias("total")).collect()[0]["total"] 
top9Total = top9Companies.agg(spark_sum("count").alias("top9_total")).collect()[0]["top9_total"]

othersTotal = totalCompanyRequests - top9Total

print("\nTop 9 companies compared to others table:\n")
othersDF = spark.createDataFrame([Row(company="Other companies", count=othersTotal)])

plotDF = top9Companies.unionByName(othersDF).orderBy(col("count").desc())
plotDF.show(truncate=False)

pdf = plotDF.orderBy(col("count").desc()).toPandas()


plt.figure(figsize=(10, 6))
plt.barh(pdf["company"], pdf["count"]) 
plt.xlabel("Number of requests") 
plt.ylabel("Company")
plt.title("Top 9 UK Companies vs Other Companies by Number of Requests") 
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("Q1_BarChart.png")


# Section D
# .ac.uk heatmap

# Reuse institutionCounts from Section B
topAcademicName = institutionCounts.orderBy(col("count").desc()).collect()[0]["institution"]
print("Most active UK academic institution domain:", topAcademicName)

parsedLogs = logFile.select(
    regexp_extract("value", r"^(\S+)", 1).alias("host"),
    regexp_extract("value", r"\[(\d{2})/(\w{3})/(\d{4}):(\d{2}):(\d{2}):(\d{2})", 1).alias("day"),
    regexp_extract("value", r"\[(\d{2})/(\w{3})/(\d{4}):(\d{2}):(\d{2}):(\d{2})", 4).alias("hour")
).cache()

topAcademicLogs = parsedLogs.filter(
    col("host").contains(".ac.uk") & col("host").contains(topAcademicName)
)

heatmapCounts = topAcademicLogs.groupBy("hour", "day").count()

pdf = heatmapCounts.toPandas()
pdf["day"] = pdf["day"].astype(int)
pdf["hour"] = pdf["hour"].astype(int)

heatmapData = pdf.pivot(index="hour", columns="day", values="count").fillna(0)

heatmapData = heatmapData.reindex(index=range(0, 24), columns=range(1, 29), fill_value=0)

# Plot heatmap
fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(heatmapData, aspect="auto", origin="lower")

ax.set_title(f"Heatmap of Requests for {topAcademicName}")
ax.set_xlabel("Day of Month")
ax.set_ylabel("Hour of Day")

ax.set_xticks(range(28))
ax.set_xticklabels(range(1, 29))

ax.set_yticks(range(24))
ax.set_yticklabels(range(0, 24))

fig.colorbar(im, ax=ax, label="Number of Requests")

fig.subplots_adjust(bottom=0.18)

fig.text(
    0.5, 0.04,
    (f"Figure: Heatmap showing request intensity by day of month and hour of day for the most "
     f"active UK academic institution domain ({topAcademicName}). Institution level is defined "
     f"using the last three labels of the hostname."),
    ha="center",
    fontsize=9,
    wrap=True
)

fig.savefig("Q1_sectionD_academic_heatmap.png", bbox_inches="tight")

#Section D co.uk heatmap

topCompanyName = companyCounts.orderBy(col("count").desc()).collect()[0]["company"]
print("Most active UK company domain:", topCompanyName)

topCompanyLogs = parsedLogs.filter(col("host").contains(topCompanyName))

# Count requests by hour and day
companyHeatmapCounts = topCompanyLogs.groupBy("hour", "day").count()

pdf_company = companyHeatmapCounts.toPandas()
pdf_company["day"] = pdf_company["day"].astype(int)
pdf_company["hour"] = pdf_company["hour"].astype(int)

companyHeatmapData = pdf_company.pivot(index="hour", columns="day", values="count").fillna(0)

companyHeatmapData = companyHeatmapData.reindex(index=range(0, 24), columns=range(1, 29), fill_value=0)

fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(companyHeatmapData, aspect="auto", origin="lower")

ax.set_title(f"Heatmap of Requests for {topCompanyName}")
ax.set_xlabel("Day of Month")
ax.set_ylabel("Hour of Day")

ax.set_xticks(range(28))
ax.set_xticklabels(range(1, 29))

ax.set_yticks(range(24))
ax.set_yticklabels(range(0, 24))

fig.colorbar(im, ax=ax, label="Number of Requests")

fig.subplots_adjust(bottom=0.18)

fig.text(
    0.5, 0.04,
    (f"Figure: Heatmap showing request intensity by day of month and hour of day for the most "
     f"active UK company domain ({topCompanyName}). Company domain level is defined using the "
     f"last three labels of the hostname."),
    ha="center",
    fontsize=9,
    wrap=True
)

fig.savefig("Q1_sectionD_company_heatmap.png", bbox_inches="tight")

#Section D gov.uk heatmap

governmentParsed = parsedLogs.filter(col("host").contains(".gov.uk"))

heatmapCountsGov = governmentParsed.groupBy("hour", "day").count()

heatmapPandasGov = heatmapCountsGov.toPandas()
heatmapPandasGov["day"] = heatmapPandasGov["day"].astype(int)
heatmapPandasGov["hour"] = heatmapPandasGov["hour"].astype(int)

heatmapDataGov = heatmapPandasGov.pivot(index="hour", columns="day", values="count").fillna(0)
heatmapDataGov = heatmapDataGov.reindex(index=range(0, 24), columns=range(1, 29), fill_value=0)

fig, ax = plt.subplots(figsize=(12, 7))
im = ax.imshow(heatmapDataGov, aspect="auto", origin="lower")

ax.set_title("Heatmap of Requests for All UK Government Domains")
ax.set_xlabel("Day of Month")
ax.set_ylabel("Hour of Day")

ax.set_xticks(range(28))
ax.set_xticklabels(range(1, 29))

ax.set_yticks(range(24))
ax.set_yticklabels(range(0, 24))

fig.colorbar(im, ax=ax, label="Number of Requests")

fig.subplots_adjust(bottom=0.18)
fig.text(
    0.5, 0.04,
    ("Figure: Heatmap showing the number of requests by day of month and hour of day "
     "for all UK government hosts ending with .gov.uk combined."),
    ha="center",
    fontsize=9,
    wrap=True
)

fig.savefig("Q1_sectionD_government_heatmap.jpg", bbox_inches="tight")


print("------ END OF THE CODE -------")

spark.stop()
