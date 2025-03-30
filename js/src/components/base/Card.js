import React, { useState } from "react";
import Image from "../../images/image1.jpg";
import Button from "react-bootstrap/Button";
import "../../css/Card.scss";

const Card = (props) => {
  const initialState = { open: false };
  const [clickState, setClickState] = useState(initialState);

  const handleClick = (props) => {
    if (clickState.open) {
      setClickState({ open: false });
    } else {
      setClickState({ open: true });
    }
  };

  return (
    <>
      {/* <div
        className={
          props.name === props.selected.item.name && props.selected.active
            ? "container selected"
            : "card"
        }
      >
        <div>{props.name}</div>
        <img src={Image} className="card-image" alt="Image" />
      </div> */}

      <div
        className={"container " + (clickState.open ? "expand" : "")}
        onClick={handleClick}
      >
        <div className="upper">
          <p>{props.name}</p>
          {/* <button type="submit" onClick={handleDelete(section.id)}> */}
          <Button
            variant="primary"
            onClick={() => props.handleDelete(props.id)}
          >
            Delete
          </Button>
          <h3>
            A family saga with a supernatural twist, set in a German town, where
            the disappearance of two young children exposes
            {
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                <polygon points="16.172 9 10.101 2.929 11.515 1.515 20 10 19.293 10.707 11.515 18.485 10.101 17.071 16.172 11 0 11 0 9" />
              </svg>
            }
          </h3>
        </div>
        <div className="lower">
          <p>{props.name}</p>
          <h3>
            A family saga with a supernatural twist, set in a German town, where
            the disappearance of two young children exposes...
          </h3>
          <h4>All News</h4>
        </div>
      </div>
    </>
  );
};

export default Card;
