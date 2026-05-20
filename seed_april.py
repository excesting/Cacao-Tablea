import random
from datetime import date, timedelta
from app import app, db, DailyProduction, ProductionHistory

def generate_april_data():
    with app.app_context():
        print("🌱 Starting April 2024 Daily & Monthly Seeding...")

        # 1. Define the timeframe (April 1 to April 30, 2024)
        # Using 2024 perfectly aligns with Weeks 14-17 shown in your CSV image
        start_date = date(2024, 4, 1)
        days_in_april = 30

        # Clean up any existing April 2024 data to prevent duplicates if run multiple times
        DailyProduction.query.filter(db.extract('month', DailyProduction.date) == 4, 
                                     db.extract('year', DailyProduction.date) == 2024).delete(synchronize_session=False)
        db.session.commit()

        weekly_data = {}

        # 2. Generate Daily Records
        for i in range(days_in_april):
            current_date = start_date + timedelta(days=i)
            year, week_num, _ = current_date.isocalendar()
            week_id = f"{year}-W{week_num}"
            month_name = current_date.strftime('%B')

            # Generate realistic daily numbers based on your historical CSV
            # Weekly average was ~50,000, so daily is ~7,142
            produced = random.randint(6800, 7500)
            
            # Historical defect rate is ~2.5%
            defect_rate = random.uniform(0.02, 0.035) 
            total_defects = int(produced * defect_rate)
            
            # Split defects between cracks and fat bloom
            cracked = int(total_defects * 0.65)
            bloom = total_defects - cracked
            good = produced - total_defects

            # Add to Daily Production table
            daily_log = DailyProduction(
                date=current_date,
                week_id=week_id,
                month_name=month_name,
                total_produced=produced,
                total_defects=total_defects,
                cracked_count=cracked,
                bloom_count=bloom,
                good_count=good,
                net_usable_output=good
            )
            db.session.add(daily_log)

            # Aggregate for the weekly rollup
            if week_id not in weekly_data:
                weekly_data[week_id] = {
                    'produced': 0, 'usable': 0, 'month': month_name
                }
            weekly_data[week_id]['produced'] += produced
            weekly_data[week_id]['usable'] += good

        db.session.commit()
        print(f"✅ Created 30 daily logs for April.")

        # 3. Generate the Weekly Master Records (ProductionHistory)
        for week_id, totals in weekly_data.items():
            # Check if this week already exists to update it, or create new
            history = ProductionHistory.query.filter_by(week_id=week_id).first()
            
            w_produced = totals['produced']
            w_usable = totals['usable']
            w_defect_rate = ((w_produced - w_usable) / w_produced) if w_produced > 0 else 0.0
            
            # Simulate sales matching roughly 98% of usable output
            w_sales = int(w_usable * random.uniform(0.95, 0.99))

            if not history:
                max_idx = db.session.query(db.func.max(ProductionHistory.time_idx)).scalar() or 0
                history = ProductionHistory(
                    time_idx=max_idx + 1, 
                    branch="Lipa", 
                    month=totals['month'], 
                    week_id=week_id, 
                    event_name="None"
                )
                db.session.add(history)

            history.total_produced = w_produced
            history.net_usable_output = w_usable
            history.defect_rate = w_defect_rate
            history.sales_pcs = w_sales

        db.session.commit()
        print(f"✅ Rolled up daily data into {len(weekly_data)} Weekly Master Records.")
        print("🎉 April Seeding Complete!")

if __name__ == '__main__':
    generate_april_data()
