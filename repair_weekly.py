from app import app, db, DailyProduction, ProductionHistory
import random

def repair_weekly_records():
    with app.app_context():
        print("🔧 Rebuilding weekly records from daily_production...")

        # 1. Get all daily records, group by week_id
        all_daily = DailyProduction.query.all()
        print(f"   Found {len(all_daily)} daily records")

        weekly_data = {}
        for d in all_daily:
            wid = d.week_id
            if wid not in weekly_data:
                weekly_data[wid] = {
                    'produced': 0, 'usable': 0, 'month': d.month_name
                }
            weekly_data[wid]['produced'] += d.total_produced
            weekly_data[wid]['usable']   += d.net_usable_output

        print(f"   Grouped into {len(weekly_data)} unique weeks")

        # 2. Get current max time_idx ONCE
        max_idx = db.session.query(db.func.max(ProductionHistory.time_idx)).scalar() or 0
        print(f"   Current max time_idx: {max_idx}")

        # 3. Create/update weekly records
        created = 0
        updated = 0
        for week_id, totals in weekly_data.items():
            history = ProductionHistory.query.filter_by(week_id=week_id).first()

            w_produced = totals['produced']
            w_usable   = totals['usable']
            w_defect_rate = ((w_produced - w_usable) / w_produced) if w_produced > 0 else 0.0
            w_sales = int(w_usable * random.uniform(0.95, 0.99))

            if not history:
                max_idx += 1
                history = ProductionHistory(
                    time_idx=max_idx,
                    branch="Lipa",
                    month=totals['month'],
                    week_id=week_id,
                    event_name="None"
                )
                db.session.add(history)
                created += 1
            else:
                updated += 1

            history.total_produced    = w_produced
            history.net_usable_output = w_usable
            history.defect_rate       = w_defect_rate
            history.sales_pcs         = w_sales

        # 4. Commit with error handling
        try:
            db.session.commit()
            print(f"\n✅ SUCCESS")
            print(f"   Created: {created} new weekly records")
            print(f"   Updated: {updated} existing records")
        except Exception as e:
            db.session.rollback()
            print(f"\n❌ Commit failed: {e}")

        # 5. Verify
        april = ProductionHistory.query.filter(
            ProductionHistory.week_id.like('2026-W%')
        ).all()
        print(f"\n   April weekly records now: {len(april)}")
        for h in sorted(april, key=lambda x: x.week_id):
            print(f"     {h.week_id}: produced={h.total_produced} "
                  f"usable={h.net_usable_output} sales={h.sales_pcs}")

if __name__ == '__main__':
    repair_weekly_records()
