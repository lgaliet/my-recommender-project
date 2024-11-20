import React from 'react';

const MovieList = (props) => {
    const FavoriteComponent = props.favoriteComponent;
    
    // Ensure movies is not null or undefined
    const movies = props.movies || [];

    return (
        <>
            {movies.map((movie, index) => (
                <div className="image-container col-auto me-2 mb-3" key={index}>
                    <img src={movie.Poster} alt='movie' className="img-fluid rounded" />
                    <div onClick={() => props.handleFavoritesClick(movie)} className='overlay d-flex align-items-center justify-content-center'>
                        <FavoriteComponent />
                    </div>
                </div>
            ))}
        </>
    );
};

export default MovieList;
