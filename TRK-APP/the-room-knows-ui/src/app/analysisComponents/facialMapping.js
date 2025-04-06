"use client";

import React, { useRef, useState, useEffect } from "react";
import Webcam from "react-webcam";
import axios from "axios";
import "bootstrap/dist/css/bootstrap.min.css";
import withAuth from "../../../hoc/withAuth";
import { useRouter } from "next/navigation";

const FacialMapping = () => {
    return (
        <>
            <h1>Facial Feature Analysis</h1>
            <button>Capture and Analyze</button>
        </>
    );
}