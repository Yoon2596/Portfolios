from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from sqlalchemy import func
from models import db, Web
import pymysql
pymysql.install_as_MySQLdb()

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://kita3:kita3@localhost:3306/kita3_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = '키 필요'

db.init_app(app)
migrate = Migrate(app, db)

ALLOWED_LOCATIONS = ['강남구', '마포구', '용산구', '성동구']
ALLOWED_FOOD_TYPES = ['다이닝바', '중식', '카페', '양식', '한식', '일식', '코스요리', '기타', '주류', '육류', '뷔페','아시아음식','퓨전']
RESULTS_PER_PAGE = 12


@app.route('/')
def index():
    locations = ALLOWED_LOCATIONS
    food_types= ALLOWED_FOOD_TYPES 
    results = db.session.query(Web.id, Web.ShopName, Web.ShopRating, Web.Reviews, Web.Address, Web.Address, Web.Category_catchtable, Web.LunchDinner, Web.Description).order_by(func.random()).limit(10).all()
    return render_template('search.html', locations=locations, food_types=food_types, results=results)

@app.route('/search', methods=['GET', 'POST'])
def search():
    locations = ALLOWED_LOCATIONS
    location = request.args.getlist('location')
    food_types= ALLOWED_FOOD_TYPES # <== MS 추가
    food_type = request.args.getlist('food_type') # <== MS 추가
    min_price = request.args.get('min_price')
    max_price = request.args.get('max_price')
    meal_choice = request.args.get('meal_choice')
    page = request.args.get('page', 1, type=int)
    sort_by = request.args.get('sort_by')

    query = db.session.query(Web)

    if location:
        subquery = db.session.query(func.min(Web.id).label('min_id')).group_by(Web.ShopName).subquery()
        query = query.filter(Web.id.in_(subquery)).filter(Web.Location.in_(location))
    
    if food_type:
        subquery = db.session.query(func.min(Web.id).label('min_id')).group_by(Web.ShopName).subquery()
        query = query.filter(Web.id.in_(subquery)).filter(Web.Category_catchtable_1.in_(food_type))
    
    if min_price and max_price:
        if meal_choice == '점심':
            subquery = db.session.query(func.min(Web.id).label('min_id')).filter(
                Web.min_점심 != '영업안함',
                Web.max_점심 != '영업안함',
                Web.avg_점심 != '영업안함',
                Web.min_점심.between(int(min_price), int(max_price)) |
                Web.max_점심.between(int(min_price), int(max_price)) |
                Web.avg_점심.between(int(min_price), int(max_price))
            ).group_by(Web.ShopName).subquery()
            query = query.filter(Web.id.in_(subquery))
        elif meal_choice == '저녁':
            subquery = db.session.query(func.min(Web.id).label('min_id')).filter(
                Web.min_저녁 != '영업안함',
                Web.max_저녁 != '영업안함',
                Web.avg_저녁 != '영업안함',
                Web.min_저녁.between(int(min_price), int(max_price)) |
                Web.max_저녁.between(int(min_price), int(max_price)) |
                Web.avg_저녁.between(int(min_price), int(max_price))
            ).group_by(Web.ShopName).subquery()
            query = query.filter(Web.id.in_(subquery))

    if sort_by == 'location': # (0718 MS 수정)
        query = query.order_by(Web.Location, Web.Category_catchtable_1)
    elif sort_by == 'food_type': # (0718 MS 수정)
        query = query.order_by(Web.Category_catchtable_1.desc(), Web.Location.desc())
    elif sort_by == 'rating':
        query = query.order_by(Web.ShopRating.desc())
    elif sort_by == 'bookmark':
        query = query.order_by(Web.Save.desc()) 
    elif sort_by == 'reviews':
        query = query.order_by(Web.Reviews.desc())

    results = query.paginate(page=page, per_page=RESULTS_PER_PAGE)

    return render_template('result.html', results=results, locations=locations, location=location,
                           food_types=food_types, food_type=food_type,
                           min_price=min_price, max_price=max_price, meal_choice=meal_choice, sort_by=sort_by)

@app.route('/save', methods=['POST'])
def save():
    card_id = request.form.get('card_id')
    if 'save_list' not in session:
        session['save_list'] = []
    if card_id not in session['save_list']:
        session['save_list'].append(card_id)
        session.modified = True
    return redirect(request.referrer)

@app.route('/unsave', methods=['POST'])
def unsave():
    card_id = request.form.get('card_id')
    if 'save_list' in session and card_id in session['save_list']:
        session['save_list'].remove(card_id)
        session.modified = True
    return redirect(request.referrer)

@app.route('/saved')
def saved():
    if 'save_list' in session:
        save_list = session['save_list']
        saved_results = db.session.query(Web).filter(Web.id.in_(save_list)).all()
    else:
        saved_results = []
    return render_template('saved.html', saved_results=saved_results)

@app.route('/map/<int:id>')
def map_view(id):
    shop = Web.query.get(id)
    if shop:
        address = shop.Address
        shopName = shop.ShopName
        shopRate = shop.ShopRating
        reviews = shop.Reviews
        price = shop.LunchDinner
        category = shop.Category_catchtable
        description = shop.Description

        # Fetch reviews for the shop using ShopName
        user_reviews = db.session.query(Web.Nickname, Web.User_Rating, Web.MealTime, Web.Date, Web.User_Review).filter_by(ShopName=shopName).all()

        # Transform reviews to a list of dictionaries
        user_reviews_list = [{
            "Nickname": review.Nickname,
            "User_Rating": review.User_Rating,
            "MealTime": review.MealTime,
            "Date": review.Date,
            "User_Review": review.User_Review
        } for review in user_reviews]

        subquery = db.session.query(func.min(Web.id).label('min_id')).group_by(Web.ShopName).subquery()
        nearby_shops = db.session.query(Web).filter(
            Web.id.in_(subquery),
            Web.Location == shop.Location,
            Web.Category_catchtable_1.in_(['주류', '카페', '다이닝바'])
        ).all()



        return render_template('map.html', shopName=shopName, address=address, shopRate=shopRate, reviews=reviews,
                               price=price, description=description, category=category, user_reviews=user_reviews_list, nearby_shops=nearby_shops)
    return redirect(url_for('saved'))


if __name__ == '__main__':
    with app.app_context():
        app.run(debug=True)
