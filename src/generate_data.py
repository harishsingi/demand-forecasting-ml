import csv, random, pathlib, datetime, math
random.seed(7)
def main():
    pathlib.Path("data").mkdir(exist_ok=True)
    with open("data/demand.csv","w",newline="") as f:
        wr=csv.writer(f); wr.writerow(["date","part","region","demand"])
        parts={"brake pad":100,"filter":160,"battery":80}
        for part,base in parts.items():
            for i in range(540):
                d=datetime.date(2023,1,1)+datetime.timedelta(days=i)
                seasonal=1+0.2*math.sin(i/30.0)
                wr.writerow([d.isoformat(),part,"SE",max(0,round(base*seasonal+random.gauss(0,10)))])
    print("ok")
if __name__=="__main__": main()
