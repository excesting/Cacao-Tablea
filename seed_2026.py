"""
seed_2026.py — Seed continuous 2026 production data (January through April).

Creates daily_production rows AND weekly production_history rollups
for every week from Jan 1 to Apr 30, 2026 — giving the forecast model
a clean, gapless block of recent weeks to work with.

IDEMPOTENT: safe to run more than once. It deletes only the 2026
daily rows it manages, then rebuilds cleanly. It does NOT touch
your historical Hist-W records.

Run locally:  python seed_2026.py
(or set DATABASE_URL to point at Railway, then run)
"""
import random
from datetime import date, timedelta
from app import app, db, DailyProduction, ProductionHistory


def seed_2026():
    with app.app_context():
        print("🌱 Seeding 2026 data: January → April...")

        start_date = date(2026, 1, 1)
        end_date   = date(2026, 4, 30)

        # ---- 1. Clean existing 2026 daily rows (so re-runs are safe) ----
        deleted = DailyProduction.query.filter(
            db.extract('year', DailyProduction.date) == 2026
        ).delete(synchronize_session=False)
        db.session.commit()
        print(f"   Cleared {deleted} existing 2026 daily rows")

        # ---- 2. Generate daily rows for every day Jan 1 → Apr 30 ----
        weekly_data = {}
        current = start_date
        day_count = 0

        while current <= end_date:
            year, week_num, _ = current.isocalendar()
            week_id    = f"{year}-W{week_num}"
            month_name = current.strftime('%B')

            produced     = random.randint(6800, 7500)
            defect_rate  = random.uniform(0.02, 0.035)
            total_defects = int(produced * defect_rate)
            cracked = int(total_defects * 0.65)
            bloom   = total_defects - cracked
            good    = produced - total_defects

            db.session.add(DailyProduction(
                date=current,
                week_id=week_id,
                month_name=month_name,
                total_produced=produced,
                total_defects=total_defects,
                cracked_count=cracked,
                bloom_count=bloom,
                good_count=good,
                net_usable_output=good,
            ))

            if week_id not in weekly_data:
                weekly_data[week_id] = {
                    'produced': 0, 'usable': 0, 'month': month_name
                }
            weekly_data[week_id]['produced'] += produced
            weekly_data[week_id]['usable']   += good

            current += timedelta(days=1)
            day_count += 1

        db.session.commit()
        print(f"   Created {day_count} daily rows across {len(weekly_data)} weeks")

        # ---- 3. Build/refresh weekly production_history records ----
        # Start time_idx right after the historical records
        max_idx = db.session.query(db.func.max(ProductionHistory.time_idx)).scalar() or 0

        created = 0
        updated = 0
        for week_id, t in sorted(weekly_data.items()):
            history = ProductionHistory.query.filter_by(week_id=week_id).first()

            w_produced = t['produced']
            w_usable   = t['usable']
            w_defect_rate = ((w_produced - w_usable) / w_produced) if w_produced > 0 else 0.0
            w_sales = int(w_usable * random.uniform(0.95, 0.99))

            if not history:
                max_idx += 1
                history = ProductionHistory(
                    time_idx=max_idx,
                    branch="Lipa",
                    month=t['month'],
                    week_id=week_id,
                    event_name="None",
                )
                db.session.add(history)
                created += 1
            else:
                updated += 1

            history.total_produced    = w_produced
            history.net_usable_output = w_usable
            history.defect_rate       = w_defect_rate
            history.sales_pcs         = w_sales

        db.session.commit()
        print(f"   Weekly records: {created} created, {updated} updated")

        # ---- 4. Verify ----
        weeks_2026 = ProductionHistory.query.filter(
            ProductionHistory.week_id.like('2026-W%')
        ).order_by(ProductionHistory.time_idx).all()

        print(f"\n✅ Done. {len(weeks_2026)} weekly 2026 records now exist:")
        for h in weeks_2026:
            print(f"     time_idx={h.time_idx}  {h.week_id}  "
                  f"produced={h.total_produced}  sales={h.sales_pcs}  "
                  f"defect={h.defect_rate:.3f}")


if __name__ == '__main__':
    seed_2026()
