# p1 2mongo
import pymongo
import csv

if __name__ == '__main__':
    client = pymongo.MongoClient("mongodb://localhost:27017/admin")
    db = client["star_database"]
    collection = db["star_info"]

    with open('客户星级.csv', 'r', encoding='UTF-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            collection.insert_one(row)
