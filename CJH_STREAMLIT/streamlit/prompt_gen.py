def generate_prompt(movie_info):
    title = movie_info.get('Title')
    genre = movie_info.get('Genre')
    plot = movie_info.get('Plot')
    rating = movie_info.get('imdbRating')
    director = movie_info.get('Director')
    actors = movie_info.get('Actors')
    country = movie_info.get('Country')

    prompt = (f"프롬프트 생성 템플릿 : '{title}'\n\n"
              f"줄거리 : {plot}\n\n"
              f"장르 : {genre}\n\n"
              f"평점 : {rating}\n\n"
              f"감독 : {director}\n\n"
              f"출연진 : {actors}\n\n"
              f"리뷰 : {reviews}\n\n"
              f"스타일 : 영화 포스터를 만들기 위한 프롬프트를 생성해주세요.")
    
    return prompt