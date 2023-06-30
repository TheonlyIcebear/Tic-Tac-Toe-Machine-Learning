import React, { useState, useEffect } from "react";
import axios from 'axios'

const [response, setResponse] = useState(null);
const [error, setError] = useState("");
const [loading, setloading] = useState(true);

export const Request = async( body ) => {

  const fetchData = async (body) => {
    axios.post("https://api.ai-tic-tac-toe.repl.co/predict", body
    ).then((res) => {
        setResponse(res.data);
      })
      .catch((err) => {
        setError(err);
      })
      .finally(() => {
        setloading(false);
      });
  };

  useEffect(() => {
    fetchData();
  }, [url, body]);

  return { response };
};

export default Request;