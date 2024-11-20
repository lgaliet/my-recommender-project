import React from 'react';

const SearchBox = (props) => {
    const handleKeyPress = (event) => {
        if (event.key === 'Enter') {
            props.onSearch();
        }
    }

    return (
        <div className="col col-sm-4">
            <input 
               className='form-control' 
               value={props.value}
               onChange={(event)=>props.setSearchValue(event.target.value)}
               onKeyPress={handleKeyPress}
               placeHolder='Type to search'
            ></input>
        </div>
    );
};

export default SearchBox;