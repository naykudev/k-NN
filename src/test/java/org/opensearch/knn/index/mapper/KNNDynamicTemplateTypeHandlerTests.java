/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.mapper;

import org.opensearch.common.xcontent.LoggingDeprecationHandler;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.index.mapper.FieldValueParserSupplier;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class KNNDynamicTemplateTypeHandlerTests extends KNNTestCase {

    private final KNNDynamicTemplateTypeHandler handler = new KNNDynamicTemplateTypeHandler();

    /** Produces a parser positioned at the array value; fails the test if a "no parser expected" case opens it. */
    private FieldValueParserSupplier arrayParserFactory(int dimension) {
        return () -> {
            StringBuilder sb = new StringBuilder("{\"vec\":[");
            for (int i = 0; i < dimension; i++) {
                if (i > 0) {
                    sb.append(",");
                }
                sb.append("0.1");
            }
            sb.append("]}");
            XContentParser parser = XContentHelper.createParser(
                NamedXContentRegistry.EMPTY,
                LoggingDeprecationHandler.INSTANCE,
                new BytesArray(sb.toString()),
                MediaTypeRegistry.JSON
            );
            parser.nextToken(); // START_OBJECT
            parser.nextToken(); // FIELD_NAME
            parser.nextToken(); // START_ARRAY
            return parser;
        };
    }

    private FieldValueParserSupplier failingParserFactory() {
        return () -> { throw new AssertionError("handler must not open the parser for a complete config"); };
    }

    public void testInjectsDimensionFromArrayLength() throws IOException {
        Map<String, Object> config = new HashMap<>();
        config.put("type", KNNVectorFieldMapper.CONTENT_TYPE);
        handler.adjustMappingConfig(config, arrayParserFactory(128));
        assertEquals(128, config.get(KNNConstants.DIMENSION));
    }

    public void testDimensionPresentOpensNoParser() throws IOException {
        Map<String, Object> config = new HashMap<>();
        config.put("type", KNNVectorFieldMapper.CONTENT_TYPE);
        config.put(KNNConstants.DIMENSION, 64);
        handler.adjustMappingConfig(config, failingParserFactory());
        assertEquals(64, config.get(KNNConstants.DIMENSION));
    }

    public void testModelIdPresentOpensNoParserAndInjectsNoDimension() throws IOException {
        Map<String, Object> config = new HashMap<>();
        config.put("type", KNNVectorFieldMapper.CONTENT_TYPE);
        config.put(KNNConstants.MODEL_ID, "my-model");
        handler.adjustMappingConfig(config, failingParserFactory());
        assertFalse("dimension must not be injected when a model supplies it", config.containsKey(KNNConstants.DIMENSION));
    }

    public void testIsConfigCompleteWithDimension() {
        Map<String, Object> config = new HashMap<>();
        config.put("type", KNNVectorFieldMapper.CONTENT_TYPE);
        config.put(KNNConstants.DIMENSION, 128);
        assertTrue(handler.isConfigComplete(config));
    }

    public void testIsConfigCompleteWithModelId() {
        Map<String, Object> config = new HashMap<>();
        config.put("type", KNNVectorFieldMapper.CONTENT_TYPE);
        config.put(KNNConstants.MODEL_ID, "my-model");
        assertTrue(handler.isConfigComplete(config));
    }

    public void testIsConfigIncompleteWithoutDimensionOrModel() {
        Map<String, Object> config = new HashMap<>();
        config.put("type", KNNVectorFieldMapper.CONTENT_TYPE);
        assertFalse(handler.isConfigComplete(config));
    }

    public void testNonArrayValueInjectsNoDimension() throws IOException {
        Map<String, Object> config = new HashMap<>();
        config.put("type", KNNVectorFieldMapper.CONTENT_TYPE);
        FieldValueParserSupplier scalarFactory = () -> {
            XContentParser parser = XContentHelper.createParser(
                NamedXContentRegistry.EMPTY,
                LoggingDeprecationHandler.INSTANCE,
                new BytesArray("{\"vec\":\"not-an-array\"}"),
                MediaTypeRegistry.JSON
            );
            parser.nextToken(); // START_OBJECT
            parser.nextToken(); // FIELD_NAME
            parser.nextToken(); // VALUE_STRING
            return parser;
        };
        handler.adjustMappingConfig(config, scalarFactory);
        assertFalse(config.containsKey(KNNConstants.DIMENSION));
    }
}
